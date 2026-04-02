#!/usr/bin/env python3
"""
stage5_second_screen.py
二筛：
  1. 从 risk_eval 读取风险 / 估值标签
  2. 过滤掉 高风险 或 高估值 的股票
  3. 从 model_predict/scores.csv 补充 score / rank / rank_pct /
     annualized_return / max_drawdown / sharpe_ratio / monthly_win_rate
  4. 按 rank 升序（得分最高）取 hold_num 支
输出: <output>/second_screen.csv
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def second_screen(risk_eval_csv: str, pred_dir: str,
                  output_dir: str, hold_num: int = 5):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载风险评估结果 ───────────────────────
    risk_df = pd.read_csv(risk_eval_csv)
    if "code" not in risk_df.columns and "instrument" in risk_df.columns:
        risk_df["code"] = risk_df["instrument"]
    print(f"✓ 加载 risk_eval: {len(risk_df)} 只股票")

    # ── 2. 过滤 高风险 / 高估值 ───────────────────
    before = len(risk_df)
    mask_ok = (
        (risk_df["risk_label"] != "高风险") &
        (risk_df["valuation_label"] != "高估值")
    )
    filtered = risk_df[mask_ok].copy()
    print(f"✓ 过滤高风险/高估值: {before} → {len(filtered)} 只")

    if filtered.empty:
        print("  ⚠ 过滤后无剩余股票！将保留中风险/正常估值股票（放宽条件）")
        filtered = risk_df[risk_df["risk_label"] != "高风险"].copy()

    # ── 3. 从 model_predict/scores.csv 补充得分信息 ─
    scores_csv = Path(pred_dir) / "scores.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(f"scores.csv 不存在: {scores_csv}")
    scores_df = pd.read_csv(scores_csv)

    # 统一 code 字段
    if "code" not in scores_df.columns and "instrument" in scores_df.columns:
        scores_df["code"] = scores_df["instrument"].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)

    merge_cols = ["code", "score", "rank", "rank_pct"]
    perf_cols  = ["annualized_return", "max_drawdown", "sharpe_ratio", "ICIR", "monthly_win_rate"]
    for c in perf_cols:
        if c in scores_df.columns:
            merge_cols.append(c)

    # 如果 filtered 里已有 score 列就先丢掉，避免重复
    drop_cols = [c for c in merge_cols if c in filtered.columns and c != "code"]
    filtered  = filtered.drop(columns=drop_cols, errors="ignore")

    result = filtered.merge(
        scores_df[merge_cols],
        on="code",
        how="left"
    )

    # ── 4. 按 rank 升序取 top hold_num ───────────
    if "rank" in result.columns:
        result = result.sort_values("rank").head(hold_num).reset_index(drop=True)
    else:
        result = result.sort_values("score", ascending=False).head(hold_num).reset_index(drop=True)

    print(f"\n◆ 二筛完成，选出 {len(result)} 只股票:")
    display_cols = ["code", "score", "rank", "rank_pct",
                    "valuation_label", "risk_label",
                    "annualized_return", "max_drawdown", "sharpe_ratio", "ICIR", "monthly_win_rate"]
    display_cols = [c for c in display_cols if c in result.columns]
    print(result[display_cols].to_string(index=False))

    out_csv = out_path / "second_screen.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ 二筛结果保存: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-eval", required=True, dest="risk_eval")
    ap.add_argument("--pred-dir",  required=True, dest="pred_dir")
    ap.add_argument("--output",    required=True)
    ap.add_argument("--hold-num",  type=int, default=5, dest="hold_num")
    args = ap.parse_args()
    second_screen(args.risk_eval, args.pred_dir, args.output, args.hold_num)


if __name__ == "__main__":
    main()
