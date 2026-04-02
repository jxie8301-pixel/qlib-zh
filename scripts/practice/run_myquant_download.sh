#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/DATA/myquant}"
INDEX_CODE="${INDEX_CODE:-SHSE.000300}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-$(date +%F)}"
HISTORY_DAYS="${HISTORY_DAYS:-0}"

if [[ -z "${GM_TOKEN:-}" ]]; then
  echo "GM_TOKEN is required in the environment before running this script." >&2
  exit 2
fi

docker run --rm \
  --platform linux/amd64 \
  -e GM_TOKEN \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  python:3.11-slim bash -lc "
    set -euo pipefail
    python -m pip install --no-cache-dir -q -i https://pypi.org/simple gm pandas numpy >/tmp/gm_install.log 2>&1 || { cat /tmp/gm_install.log; exit 1; }
    python scripts/practice/download_myquant_data.py \
      --output '$OUTPUT_DIR' \
      --index '$INDEX_CODE' \
      --start-date '$START_DATE' \
      --end-date '$END_DATE' \
      --history-days '$HISTORY_DAYS'
  "