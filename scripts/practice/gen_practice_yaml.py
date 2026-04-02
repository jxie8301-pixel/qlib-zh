#!/usr/bin/env python3
"""
gen_practice_yaml.py
根据动态日期范围，从模板 YAML 生成本次实盘实验专用的 workflow_config.yaml。
"""
import argparse
import re
import sys
from pathlib import Path

import yaml


def patch_yaml(template_path: str, output_path: str, dates: dict) -> None:
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = yaml.safe_load(content)

    # ──────────────────────────────────────────
    # 1. 更新 data_handler_config
    # ──────────────────────────────────────────
    dh = doc.get("data_handler_config", {})
    # handler 的全局时间范围覆盖训练+验证+测试
    dh["start_time"]     = dates["train_start"]
    dh["end_time"]       = dates["test_end"]
    dh["fit_start_time"] = dates["train_start"]
    dh["fit_end_time"]   = dates["train_end"]

    # ──────────────────────────────────────────
    # 2. 更新 dataset segments
    # ──────────────────────────────────────────
    segments = doc["task"]["dataset"]["kwargs"]["segments"]
    segments["train"] = [dates["train_start"], dates["train_end"]]
    segments["valid"] = [dates["valid_start"], dates["valid_end"]]
    segments["test"]  = [dates["test_start"],  dates["test_end"]]

    # 更新 handler 引用（内联 kwargs 方式兼容）
    hkw = doc["task"]["dataset"]["kwargs"]["handler"]
    if isinstance(hkw, dict) and "kwargs" in hkw:
        hkw["kwargs"].update(
            start_time=dates["train_start"],
            end_time=dates["test_end"],
            fit_start_time=dates["train_start"],
            fit_end_time=dates["train_end"],
        )

    # ──────────────────────────────────────────
    # 3. 更新 port_analysis_config backtest 时间
    # ──────────────────────────────────────────
    pa = doc.get("port_analysis_config", {})
    bt = pa.get("backtest", {})
    bt["start_time"] = dates["train_start"]
    bt["end_time"]   = dates["valid_end"]

    # ──────────────────────────────────────────
    # 4. 写出
    # ──────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"✓ YAML 已写出: {output_path}")
    print(f"  train : {dates['train_start']} → {dates['train_end']}")
    print(f"  valid : {dates['valid_start']} → {dates['valid_end']}")
    print(f"  test  : {dates['test_start']}  → {dates['test_end']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template",    required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--train-start", required=True, dest="train_start")
    ap.add_argument("--train-end",   required=True, dest="train_end")
    ap.add_argument("--valid-start", required=True, dest="valid_start")
    ap.add_argument("--valid-end",   required=True, dest="valid_end")
    ap.add_argument("--test-start",  required=True, dest="test_start")
    ap.add_argument("--test-end",    required=True, dest="test_end")
    args = ap.parse_args()

    patch_yaml(
        args.template,
        args.output,
        {
            "train_start": args.train_start,
            "train_end":   args.train_end,
            "valid_start": args.valid_start,
            "valid_end":   args.valid_end,
            "test_start":  args.test_start,
            "test_end":    args.test_end,
        },
    )


if __name__ == "__main__":
    main()
