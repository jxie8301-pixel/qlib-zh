#!/usr/bin/env python3
"""
Download recent 10 years of daily CN data into DATA/ and dump to qlib format.

Usage:
    python scripts/download_10y_data.py

Runs the collector to: download -> normalize -> dump_all (qlib bin)
"""
import sys
import subprocess
import datetime
from pathlib import Path


def run(cmd):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"Command failed with exit {r.returncode}")
        sys.exit(r.returncode)


def main():
    today = datetime.date.today()
    # ~10 years (approx 365.25*10 ~= 3652)
    start = today - datetime.timedelta(days=3652)
    start_s = start.strftime("%Y-%m-%d")
    end_s = today.strftime("%Y-%m-%d")

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root.joinpath("DATA")
    src_dir = data_root.joinpath("source")
    normalize_dir = data_root.joinpath("normalize")
    qlib_dir = data_root.joinpath("qlib_data/cn_data")

    for p in (src_dir, normalize_dir, qlib_dir):
        p.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # If the user already has qlib 1d data, prefer update flow (avoids fetching full instrument list)
    user_qlib_dir = Path("~/.qlib/qlib_data/cn_data").expanduser()
    if user_qlib_dir.exists():
        print(f"Detected existing qlib data at {user_qlib_dir}. Using update_data_to_bin (faster).")
        run([
            py,
            str(repo_root.joinpath("scripts/data_collector/yahoo/collector.py")),
            "update_data_to_bin",
            "--qlib_data_1d_dir",
            str(user_qlib_dir),
            "--end_date",
            end_s,
        ])
    else:
        # 1) download 1d CN data from Yahoo
        run([
            py,
            str(repo_root.joinpath("scripts/data_collector/yahoo/collector.py")),
            "download_data",
            "--source_dir",
            str(src_dir),
            "--normalize_dir",
            str(normalize_dir),
            "--region",
            "CN",
            "--interval",
            "1d",
            "--start",
            start_s,
            "--end",
            end_s,
            "--delay",
            "0.5",
            "--max_collector_count",
            "2",
        ])

        # 2) normalize downloaded data
        run([
            py,
            str(repo_root.joinpath("scripts/data_collector/yahoo/collector.py")),
            "normalize_data",
            "--source_dir",
            str(src_dir),
            "--normalize_dir",
            str(normalize_dir),
            "--region",
            "CN",
            "--interval",
            "1d",
            "--end_date",
            end_s,
        ])

        # 3) dump normalized csv -> qlib bin format
        run([
            py,
            str(repo_root.joinpath("scripts/dump_bin.py")),
            "dump_all",
            "--data_path",
            str(normalize_dir),
            "--qlib_dir",
            str(qlib_dir),
            "--freq",
            "day",
        ])

    print("Done. qlib data is in:", qlib_dir)


if __name__ == "__main__":
    main()
