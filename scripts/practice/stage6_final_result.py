#!/usr/bin/env python3
"""
stage6_final_result.py
最终结果整合 & 调仓建议：
  1. 从 second_screen 读取已选股票
  2. 从 model_predict/scores.csv 查找最新排名，新增"最新排名"列
  3. 保存 result.csv
  4. 若有股票最新排名在后30%，则用 second_screen 里排名更高的股票替换
  5. 保存 result_update.csv
输出:
  <output>/result.csv
  <output>/result_update.csv
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


def final_result(second_screen_csv: str, pred_dir: str,
                 output_dir: str, hold_num: int = 5,
                 bottom_pct: float = 0.30):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载二筛结果 ───────────────────────────
    ss_df = pd.read_csv(second_screen_csv)
    if "code" not in ss_df.columns and "instrument" in ss_df.columns:
        ss_df["code"] = ss_df["instrument"].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)
    print(f"✓ 加载二筛结果: {len(ss_df)} 只股票")
    print(ss_df[["code"] + [c for c in ["name", "score", "rank_pct"] if c in ss_df.columns]].to_string(index=False))

    # ── 2. 从 scores.csv 获取最新排名 ─────────────
    scores_csv = Path(pred_dir) / "scores.csv"
    if not scores_csv.exists():
        raise FileNotFoundError(f"scores.csv 不存在: {scores_csv}")
    scores_df = pd.read_csv(scores_csv)
    if "code" not in scores_df.columns and "instrument" in scores_df.columns:
        scores_df["code"] = scores_df["instrument"].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)

    total_stocks = len(scores_df)

    # 计算最新排名（基于 score 降序，rank_pct = rank/total * 100%）
    scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
    scores_df["latest_rank_idx"] = scores_df.index + 1  # 1-based
    scores_df["latest_rank_pct"] = (
        (scores_df["latest_rank_idx"] / total_stocks * 100).round(2).astype(str) + "%"
    )
    latest_map = scores_df.set_index("code")[["latest_rank_idx", "latest_rank_pct"]].to_dict("index")

    # ── 3. 拼接最新排名到二筛结果 ─────────────────
    def _get_latest_rank(code):
        return latest_map.get(str(code), {}).get("latest_rank_idx", np.nan)

    def _get_latest_rank_pct(code):
        return latest_map.get(str(code), {}).get("latest_rank_pct", "N/A")

    result = ss_df.copy()
    result["最新排名"]     = result["code"].apply(_get_latest_rank)
    result["最新排名_pct"] = result["code"].apply(_get_latest_rank_pct)

    # ── 4. 保存 result.csv ────────────────────────
    result_csv = out_path / "result.csv"
    result_scv = out_path / "result.scv"
    result.to_csv(result_csv, index=False, encoding="utf-8-sig")
    result.to_csv(result_scv, index=False, encoding="utf-8-sig")

    print(f"\n◆ result.csv 内容:")
    disp_cols = ["code"] + [c for c in ["name", "score", "rank_pct",
                                        "最新排名", "最新排名_pct",
                                        "valuation_label", "risk_label",
                                        "annualized_return", "max_drawdown",
                                        "sharpe_ratio", "ICIR", "monthly_win_rate"] if c in result.columns]
    print(result[disp_cols].to_string(index=False))
    print(f"\n✓ result.csv 保存: {result_csv}")
    print(f"✓ result.scv 保存: {result_scv}")

    # ── 5. 判断是否需要替换（最新排名在后30%）────────
    threshold_rank = int(total_stocks * (1 - bottom_pct))  # rank > threshold → 后30%
    print(f"\n─── 调仓检查（后{int(bottom_pct*100)}% 阈值 = rank > {threshold_rank} / {total_stocks}）───")

    to_replace_codes = result[result["最新排名"] > threshold_rank]["code"].tolist()

    if not to_replace_codes:
        print("  ✓ 无需调仓，所有持仓股票最新排名均未进入后30%")
        result_update = result.copy()
        result_update["调仓说明"] = "持仓不变"
    else:
        print(f"  ⚠ 以下股票最新排名进入后30%，需替换: {to_replace_codes}")

        # 从 second_screen 的完整候选池中找替补
        # second_screen 可能只有 hold_num 只 → 去 scores.csv 找初筛里
        # 原始候选按 rank 排序，跳过已持仓
        already_hold  = set(result["code"].tolist())
        candidates    = scores_df[~scores_df["code"].isin(already_hold)].copy()

        # 尝试过滤风险/估值（如果 risk_eval 数据可用）
        risk_eval_csv = Path(second_screen_csv).parent.parent / "risk_eval" / "risk_eval.csv"
        if risk_eval_csv.exists():
            re_df = pd.read_csv(risk_eval_csv)
            if "code" not in re_df.columns and "instrument" in re_df.columns:
                re_df["code"] = re_df["instrument"].astype(str).str.replace(r'^[A-Za-z]+', '', regex=True).str.zfill(6)
            # 稳健地检查列是否存在，避免 .get() 返回标量
            risk_ok = (
                re_df["risk_label"] != "高风险"
                if "risk_label" in re_df.columns
                else pd.Series(True, index=re_df.index)
            )
            val_ok = (
                re_df["valuation_label"] != "高估值"
                if "valuation_label" in re_df.columns
                else pd.Series(True, index=re_df.index)
            )
            safe_codes = set(re_df[risk_ok & val_ok]["code"].tolist())
            candidates_filtered = candidates[candidates["code"].isin(safe_codes)]
            if not candidates_filtered.empty:
                candidates = candidates_filtered

        replacements      = candidates.head(len(to_replace_codes))
        replacement_codes = replacements["code"].tolist()

        result_update = result.copy()
        result_update["调仓说明"] = ""

        for old_code, new_code in zip(to_replace_codes, replacement_codes):
            old_rank_pct = result_update.loc[result_update["code"] == old_code, "最新排名_pct"].values[0]
            new_row_data = scores_df[scores_df["code"] == new_code].iloc[0].to_dict()

            # 替换行
            idx = result_update[result_update["code"] == old_code].index[0]
            result_update.at[idx, "code"]          = new_code
            result_update.at[idx, "score"]         = new_row_data.get("score",  np.nan)
            result_update.at[idx, "rank"]           = new_row_data.get("latest_rank_idx", np.nan)
            result_update.at[idx, "rank_pct"]       = new_row_data.get("latest_rank_pct", "N/A")
            result_update.at[idx, "最新排名"]        = new_row_data.get("latest_rank_idx", np.nan)
            result_update.at[idx, "最新排名_pct"]    = new_row_data.get("latest_rank_pct", "N/A")
            result_update.at[idx, "调仓说明"]        = (
                f"卖出{old_code}(最新排名{old_rank_pct})→买入{new_code}"
            )
            print(f"  替换: 卖出 {old_code}（最新排名 {old_rank_pct}）→ 买入 {new_code}")

        # 未替换的行填充说明
        result_update.loc[result_update["调仓说明"] == "", "调仓说明"] = "继续持有"

    # ── 6. 保存 result_update.csv ─────────────────
    result_update_csv = out_path / "result_update.csv"
    result_update.to_csv(result_update_csv, index=False, encoding="utf-8-sig")

    print(f"\n◆ result_update.csv 内容:")
    disp_cols2 = ["code"] + [c for c in ["name", "score", "rank_pct",
                                          "最新排名", "最新排名_pct",
                                          "valuation_label", "risk_label",
                                          "调仓说明"] if c in result_update.columns]
    print(result_update[disp_cols2].to_string(index=False))
    print(f"\n✓ result_update.csv 保存: {result_update_csv}")

    # ── 7. 打印操盘摘要 ───────────────────────────
    print("\n═══════════════════════════════════════════════")
    print("  本周五操盘建议摘要")
    print("═══════════════════════════════════════════════")
    cash    = float(import_env("CASH_TOTAL", 20000))
    fee     = float(import_env("TX_FEE_RATE", 0.0001))
    stamp   = float(import_env("STAMP_DUTY_RATE", 0.0005))
    hold    = int(import_env("HOLD_NUM", 5))

    per_stock = cash / hold
    print(f"  账户资金: ¥{cash:,.0f}")
    print(f"  持仓数量: {hold} 支")
    print(f"  每支仓位: ¥{per_stock:,.0f}（建议）")
    print(f"  交易费率: 万分之{fee*10000:.0f} + 印花税万分之{stamp*10000:.0f}（卖出）")
    print(f"  持仓股票:")
    for _, row in result_update.iterrows():
        note = row.get("调仓说明", "")
        latest_pct = row.get("最新排名_pct", "N/A")
        print(f"    {row['code']}  最新排名={latest_pct}  {note}")
    print("═══════════════════════════════════════════════")


def import_env(key, default):
    import os
    return os.environ.get(key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-screen", required=True, dest="second_screen")
    ap.add_argument("--pred-dir",      required=True, dest="pred_dir")
    ap.add_argument("--output",        required=True)
    ap.add_argument("--hold-num",      type=int,   default=5,    dest="hold_num")
    ap.add_argument("--bottom-pct",    type=float, default=0.30, dest="bottom_pct")
    args = ap.parse_args()
    final_result(args.second_screen, args.pred_dir, args.output,
                 args.hold_num, args.bottom_pct)


if __name__ == "__main__":
    main()
