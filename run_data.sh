#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-qlib-rdagent:latest}"
HOST_QLIB_ROOT="${QLIB_HOST_DATA_DIR:-$HOME/.qlib}"
CONTAINER_WORKDIR="/work"
RUN_TAG="${1:-pit_update_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$SCRIPT_DIR/DATA/analysis_outputs/$RUN_TAG"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "run_tag=$RUN_TAG" > "$OUT_DIR/run_summary.txt"
echo "docker_image=$DOCKER_IMAGE" >> "$OUT_DIR/run_summary.txt"
echo "host_qlib_root=$HOST_QLIB_ROOT" >> "$OUT_DIR/run_summary.txt"

CONTAINER_SCRIPT=$(cat <<'EOS'
set -euo pipefail

CAL_FILE="/root/.qlib/qlib_data/cn_data/calendars/day.txt"
if [[ ! -f "$CAL_FILE" ]]; then
  echo "Missing calendar file: $CAL_FILE" >&2
  exit 1
fi

START_DATE="$(head -n 1 "$CAL_FILE")"
END_DATE="$(tail -n 1 "$CAL_FILE")"
RAW_DIR="/root/.qlib/stock_data/source/pit"
NORM_Q_DIR="/root/.qlib/stock_data/source/pit_normalized_quarterly"
NORM_A_DIR="/root/.qlib/stock_data/source/pit_normalized_annual"
PIT_SYMBOL_REGEX="${PIT_SYMBOL_REGEX:-}"
PIT_LIMIT_NUMS="${PIT_LIMIT_NUMS:-}"
mkdir -p "$RAW_DIR" "$NORM_Q_DIR" "$NORM_A_DIR"
find "$RAW_DIR" -maxdepth 1 -type f -name '*.csv' -delete
find "$NORM_Q_DIR" -maxdepth 1 -type f -name '*.csv' -delete
find "$NORM_A_DIR" -maxdepth 1 -type f -name '*.csv' -delete

echo "[2/4] 从 baostock 拉取季度财务数据：$START_DATE ~ $END_DATE"
python - <<PY 2>&1 | tee "/work/DATA/analysis_outputs/${RUN_TAG}/logs/pit_download_quarterly.log"
import importlib.util
import time
import sys
from pathlib import Path

sys.path = [p for p in sys.path if p not in ('', '/work', '/work/')]
sys.path.append('/work/scripts/data_collector/pit')
from collector import Run

source_dir = Path('/root/.qlib/stock_data/source/pit')
source_dir.mkdir(parents=True, exist_ok=True)
download_kwargs = dict(start='${START_DATE}', end='${END_DATE}', max_collector_count=1, delay=0.1)
if '${PIT_SYMBOL_REGEX}':
    download_kwargs['symbol_regex'] = '${PIT_SYMBOL_REGEX}'
if '${PIT_LIMIT_NUMS}':
    download_kwargs['limit_nums'] = int('${PIT_LIMIT_NUMS}')

attempt = 0
while True:
    attempt += 1
    print(f'[step2] download attempt {attempt}')
    for csv in source_dir.glob('*.csv'):
        csv.unlink()

    try:
        if importlib.util.find_spec('baostock') is not None:
            import baostock as bs

            bs.login()
            Run(source_dir=str(source_dir), interval='quarterly', max_workers=1).download_data(**download_kwargs)
            bs.logout()
        else:
            from qlib.tests.data import GetData

            print('[fallback] baostock unavailable, downloading official PIT package via GetData')
            GetData().qlib_data(name='qlib_data', target_dir=str(source_dir), region='pit', delete_old=False, exists_skip=True)
    except Exception as exc:
        print(f'[warn] download attempt failed: {exc}')

    if any(source_dir.glob('*.csv')):
        print(f'[step2] download success: {len(list(source_dir.glob("*.csv")))} csv files')
        break

    print('[step2] no PIT source files detected, retrying in 60s')
    time.sleep(60)
PY

echo "[3/4] 对季度数据归一化并写入 qlib_data/cn_data"
python - <<'PY'
import sys
import pandas as pd

sys.path = [p for p in sys.path if p not in ('', '/work', '/work/')]
sys.path.append('/work/scripts/data_collector/pit')
import collector

with open('/root/.qlib/qlib_data/cn_data/calendars/day.txt', 'r', encoding='utf-8') as f:
    local_calendar = [pd.Timestamp(line.strip()) for line in f if line.strip()]

collector.get_calendar_list = lambda: local_calendar
from collector import Run

Run(
    source_dir='/root/.qlib/stock_data/source/pit',
    normalize_dir='/root/.qlib/stock_data/source/pit_normalized_quarterly',
    interval='quarterly',
    max_workers=1,
).normalize_data()
PY

python - <<'PY' 2>&1 | tee "/work/DATA/analysis_outputs/${RUN_TAG}/logs/pit_dump_quarterly.log"
import sys

sys.path = [p for p in sys.path if p not in ('', '/work', '/work/')]
sys.path.append('/work/scripts')
from dump_pit import DumpPitData

DumpPitData(
    csv_path='/root/.qlib/stock_data/source/pit_normalized_quarterly',
    qlib_dir='/root/.qlib/qlib_data/cn_data',
).dump(interval='quarterly')
PY

echo "[3/4] 对年度数据归一化并写入 qlib_data/cn_data"
python - <<'PY'
import sys
import pandas as pd

sys.path = [p for p in sys.path if p not in ('', '/work', '/work/')]
sys.path.append('/work/scripts/data_collector/pit')
import collector

with open('/root/.qlib/qlib_data/cn_data/calendars/day.txt', 'r', encoding='utf-8') as f:
    local_calendar = [pd.Timestamp(line.strip()) for line in f if line.strip()]

collector.get_calendar_list = lambda: local_calendar
from collector import Run

Run(
    source_dir='/root/.qlib/stock_data/source/pit',
    normalize_dir='/root/.qlib/stock_data/source/pit_normalized_annual',
    interval='annual',
    max_workers=1,
).normalize_data()
PY

python - <<'PY' 2>&1 | tee "/work/DATA/analysis_outputs/${RUN_TAG}/logs/pit_dump_annual.log"
import sys

sys.path = [p for p in sys.path if p not in ('', '/work', '/work/')]
sys.path.append('/work/scripts')
from dump_pit import DumpPitData

DumpPitData(
    csv_path='/root/.qlib/stock_data/source/pit_normalized_annual',
    qlib_dir='/root/.qlib/qlib_data/cn_data',
).dump(interval='annual')
PY

echo "[4/4] 检查 qlib_data/cn_data 健康状况"
python /work/scripts/check_data_health.py check_data --qlib_dir /root/.qlib/qlib_data/cn_data 2>&1 | tee "/work/DATA/analysis_outputs/${RUN_TAG}/logs/data_health.log"

echo "done"
EOS
)

echo "[1/4] 启动 $DOCKER_IMAGE 容器并进入执行环境"
docker run --rm \
  -e RUN_TAG="$RUN_TAG" \
  -v "$SCRIPT_DIR:$CONTAINER_WORKDIR" \
  -v "$HOST_QLIB_ROOT:/root/.qlib" \
  -w "$CONTAINER_WORKDIR" \
  "$DOCKER_IMAGE" \
  /bin/bash -lc "$CONTAINER_SCRIPT" 2>&1 | tee "$LOG_DIR/run_data.log"

cat >> "$OUT_DIR/run_summary.txt" <<EOF
calendar_file=/root/.qlib/qlib_data/cn_data/calendars/day.txt
raw_dir=/root/.qlib/stock_data/source/pit
quarterly_normalize_dir=/root/.qlib/stock_data/source/pit_normalized_quarterly
annual_normalize_dir=/root/.qlib/stock_data/source/pit_normalized_annual
qlib_dir=/root/.qlib/qlib_data/cn_data
EOF

echo "Done. Logs saved in: $LOG_DIR"