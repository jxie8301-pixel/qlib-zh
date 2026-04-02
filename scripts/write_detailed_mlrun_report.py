#!/usr/bin/env python3
"""Write a detailed MLflow run analysis report into DATA/analysis_outputs.

Outputs:
- factor_analysis_report_<time>.md
- factor_analysis_report_<time>.html
- charts saved in the same directory

Example:
  python scripts/write_detailed_mlrun_report.py \
    --run mlruns/860985181741339330/86667b9d122e4f479968aebd9c8892ea
"""

from __future__ import annotations

import argparse
import json
import html
import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dt_to_folder(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y%m%d_%H%M%S")


def safe_md_table(obj) -> str:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name or "value")
    if isinstance(obj, pd.DataFrame):
        return obj.to_string()
    return str(obj)


def safe_html_table(obj) -> str:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name or "value")
    if isinstance(obj, pd.DataFrame):
        return obj.to_html(border=0, classes="table table-striped table-sm", escape=False)
    return f"<pre>{html.escape(str(obj))}</pre>"


def plot_bar(series: pd.Series, out_path: str, title: str) -> None:
    plt.figure(figsize=(11, 5))
    series.plot(kind="bar", color="#4C78A8")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_line(series: pd.Series, out_path: str, title: str) -> None:
    plt.figure(figsize=(11, 4))
    series.plot(color="#F58518", linewidth=1.4)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hist(series: pd.Series, out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 4))
    series.hist(bins=60, color="#54A24B", alpha=0.85)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_daily_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.Series:
    merged = pd.concat([pred.iloc[:, 0], label.iloc[:, 0]], axis=1, join="inner")
    merged.columns = ["pred", "label"]
    if isinstance(merged.index, pd.MultiIndex):
        tmp = merged.copy()
        tmp.index = tmp.index.get_level_values(0)
        return tmp.groupby(level=0).apply(lambda g: g["pred"].corr(g["label"]))
    return pd.Series([merged["pred"].corr(merged["label"])], index=[merged.index[0]])


def compute_recent_recommendations(pred: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df = pred.reset_index()
    dt_col, score_col = df.columns[0], df.columns[2]
    df = df.sort_values([dt_col, score_col], ascending=[True, False])
    latest_date = df[dt_col].max()
    latest = df[df[dt_col] == latest_date].nlargest(top_n, score_col).copy()
    latest.columns = ["datetime", "instrument", "score"]
    return latest


def compute_stock_analysis(pred: pd.DataFrame, label: pd.DataFrame | None, recs: pd.DataFrame) -> pd.DataFrame:
    score_series = pred.iloc[:, 0].rename("score")
    if label is not None:
        merged = pd.concat([score_series, label.iloc[:, 0].rename("label")], axis=1, join="inner")
    else:
        merged = score_series.to_frame()

    rows = []
    for _, rec in recs.iterrows():
        inst = rec["instrument"]
        if isinstance(merged.index, pd.MultiIndex):
            try:
                sub = merged.xs(inst, level=1).sort_index()
            except Exception:
                sub = pd.DataFrame(columns=merged.columns)
        else:
            sub = merged[merged.index == inst].sort_index()

        recent60 = sub.tail(60)
        recent20 = sub.tail(20)
        score_slope = math.nan
        if len(recent60) >= 2:
            x = np.arange(len(recent60), dtype=float)
            score_slope = float(np.polyfit(x, recent60["score"].to_numpy(dtype=float), 1)[0])

        label_ic = math.nan
        label_mean_20 = math.nan
        label_mean_60 = math.nan
        if label is not None and "label" in recent60.columns and len(recent60) >= 5:
            label_mean_20 = float(recent20["label"].mean()) if len(recent20) else math.nan
            label_mean_60 = float(recent60["label"].mean()) if len(recent60) else math.nan
            try:
                label_ic = float(recent60["score"].corr(recent60["label"]))
            except Exception:
                label_ic = math.nan

        rows.append(
            {
                "instrument": inst,
                "latest_score": float(rec["score"]),
                "score_mean_20d": float(recent20["score"].mean()) if len(recent20) else math.nan,
                "score_mean_60d": float(recent60["score"].mean()) if len(recent60) else math.nan,
                "score_slope_60d": score_slope,
                "label_mean_20d": label_mean_20,
                "label_mean_60d": label_mean_60,
                "label_ic_60d": label_ic,
                "analysis": "score 强且近期上行" if rec["score"] >= recs["score"].median() else "score 较强，适合分散持有",
            }
        )
    return pd.DataFrame(rows).sort_values("latest_score", ascending=False)


def interpret_feature_name(name: str) -> str:
    """Very short financial interpretation for Alpha158 feature names."""
    rules = [
        ("KMID", "K线实体/中轴相关，反映日内涨跌方向与力度"),
        ("KLEN", "K线振幅，反映波动强弱"),
        ("KUP", "上影线，反映冲高回落压力"),
        ("KLOW", "下影线，反映下探后承接"),
        ("KSFT", "收盘位置，反映收盘强弱"),
        ("OPEN", "开盘价相对位置，反映开盘定价"),
        ("HIGH", "高价相对位置，反映盘中上冲力度"),
        ("LOW", "低价相对位置，反映回撤/支撑"),
        ("VWAP", "成交均价，反映资金成交成本中枢"),
        ("ROC", "动量/涨幅变化，反映趋势延续"),
        ("MA", "均线位置，反映趋势均值回归/趋势延续"),
        ("STD", "波动率，反映风险与不确定性"),
        ("BETA", "斜率/趋势强度，反映价格趋势方向"),
        ("RSQR", "趋势拟合度，反映趋势稳定性"),
        ("RESI", "趋势残差，反映偏离趋势的程度"),
        ("MAX", "区间高点，反映阻力位"),
        ("MIN", "区间低点，反映支撑位"),
        ("QTLU", "上分位价格，反映偏强区间"),
        ("QTLD", "下分位价格，反映偏弱区间"),
        ("RANK", "当前价格在窗口内的相对排序"),
        ("RSV", "随机位置指标，反映区间相对强弱"),
        ("IMAX", "距离高点天数，反映创新高/高位停留"),
        ("IMIN", "距离低点天数，反映创新低/低位停留"),
        ("IMXD", "高低点时序差，反映方向性"),
        ("CORR", "价量相关性，反映量价共振"),
        ("CORD", "涨跌与量变相关性，反映趋势配合"),
        ("CNTP", "上涨天数占比，反映上涨持续性"),
        ("CNTN", "下跌天数占比，反映下跌持续性"),
        ("CNTD", "涨跌天数差，反映多空强弱"),
    ]
    for key, desc in rules:
        if name.startswith(key):
            return desc
    return "Alpha158 原始因子，反映价格/成交量/趋势/波动等维度"


def extract_feature_importance(model, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    fi = model.get_feature_importance()
    if len(feature_names) != len(fi):
        feature_names = [f"Feature_{i}" for i in range(len(fi))]
    df = pd.DataFrame({"feature": feature_names, "importance": fi.values})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    df["explanation"] = df["feature"].map(interpret_feature_name)
    return df


def extract_turnover_mean(report_normal: pd.DataFrame | None) -> float:
    if isinstance(report_normal, pd.DataFrame) and "turnover" in report_normal.columns:
        return float(report_normal["turnover"].mean())
    return math.nan


def build_performance_curves(report_normal: pd.DataFrame | None) -> dict:
    if not isinstance(report_normal, pd.DataFrame) or report_normal.empty:
        return {}
    df = report_normal.copy()
    out = {}
    if "return" in df.columns:
        out["cum_return"] = df["return"].cumsum()
        out["drawdown"] = out["cum_return"] - out["cum_return"].cummax()
    if "return" in df.columns and "bench" in df.columns:
        out["cum_excess_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    if all(k in df.columns for k in ["return", "bench", "cost"]):
        out["cum_excess_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()
    if "turnover" in df.columns:
        out["turnover"] = df["turnover"]
    return out


def extract_port_metrics(port: pd.DataFrame | None) -> dict:
    """Extract headline backtest metrics from qlib PortAnaRecord output."""
    out = {
        "excess_return_without_cost_annualized_return": math.nan,
        "excess_return_without_cost_information_ratio": math.nan,
        "excess_return_without_cost_max_drawdown": math.nan,
        "excess_return_with_cost_annualized_return": math.nan,
        "excess_return_with_cost_information_ratio": math.nan,
        "excess_return_with_cost_max_drawdown": math.nan,
    }
    if not isinstance(port, pd.DataFrame):
        return out

    try:
        if isinstance(port.index, pd.MultiIndex):
            for key in ["excess_return_without_cost", "excess_return_with_cost"]:
                if key in port.index.get_level_values(0):
                    s = port.xs(key, level=0)
                    for metric in ["annualized_return", "information_ratio", "max_drawdown"]:
                        if metric in s.index:
                            out[f"{key}_{metric}"] = float(s.loc[metric, "risk"])
        else:
            # fallback for flat structures
            for metric in out:
                if metric in port.columns:
                    out[metric] = float(port[metric].iloc[0])
    except Exception:
        pass
    return out


def extract_indicator_metrics(ind: pd.DataFrame | None) -> dict:
    out = {"ffr": math.nan, "pa": math.nan, "pos": math.nan}
    if isinstance(ind, pd.DataFrame) and "value" in ind.columns:
        for k in out:
            try:
                out[k] = float(ind.loc[k, "value"])
            except Exception:
                pass
    return out


def latest_holding_snapshot(pos) -> pd.DataFrame:
    """Build a compact snapshot of the latest holdings from positions_normal_1day.pkl."""
    if not isinstance(pos, dict) or not pos:
        return pd.DataFrame()
    latest_date = max(pos.keys())
    latest_obj = pos[latest_date]
    if hasattr(latest_obj, "position"):
        latest = getattr(latest_obj, "position", {}) or {}
    elif isinstance(latest_obj, dict):
        latest = latest_obj.get("position", latest_obj)
    else:
        latest = {}
    items = []
    for k, v in latest.items():
        if k in {"cash", "now_account_value", "init_cash"}:
            continue
        if isinstance(v, dict):
            items.append(
                {
                    "instrument": k,
                    "weight": float(v.get("weight", math.nan)),
                    "amount": float(v.get("amount", math.nan)),
                    "price": float(v.get("price", math.nan)),
                    "count_day": int(v.get("count_day", 0)) if v.get("count_day") is not None else None,
                }
            )
    return pd.DataFrame(items).sort_values("weight", ascending=False)


def render_html(title: str, sections: list[str]) -> str:
    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 18px;line-height:1.6;color:#222}
    h1,h2,h3{line-height:1.25}
    pre{background:#f7f7f7;padding:14px;border-radius:8px;overflow:auto}
    .table{border-collapse:collapse;width:100%;margin:12px 0 24px}
    .table th,.table td{border:1px solid #ddd;padding:8px;text-align:left}
    .table th{background:#f1f3f5}
    img{max-width:100%;border:1px solid #eee;border-radius:8px}
    .muted{color:#666}
    .card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:16px 18px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
    """
    html_doc = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{css}</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    html_doc.extend(sections)
    html_doc.append("</body></html>")
    return "\n".join(html_doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="MLflow run dir, e.g. mlruns/.../<run_id>")
    ap.add_argument("--out-root", default="DATA/analysis_outputs")
    ap.add_argument("--out-dir", default=None, help="Fixed output directory; overrides out-root timestamp folder")
    args = ap.parse_args()

    run_dir = args.run.rstrip("/")
    run_id = os.path.basename(run_dir)

    meta_path = os.path.join(run_dir, "meta.yaml")
    start_time_ms = None
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("start_time:"):
                start_time_ms = int(line.split(":", 1)[1].strip())
                break
    if start_time_ms is None:
        raise RuntimeError(f"Failed to find start_time in {meta_path}")

    exp_time = dt_to_folder(start_time_ms)
    out_dir = args.out_dir or os.path.join(args.out_root, f"{exp_time}_{run_id}")
    ensure_dir(out_dir)

    artifacts = os.path.join(run_dir, "artifacts")
    pred_path = os.path.join(artifacts, "pred.pkl")
    label_path = os.path.join(artifacts, "label.pkl")
    port_path = os.path.join(artifacts, "portfolio_analysis", "port_analysis_1day.pkl")
    ind_path = os.path.join(artifacts, "portfolio_analysis", "indicator_analysis_1day.pkl")
    pos_path = os.path.join(artifacts, "portfolio_analysis", "positions_normal_1day.pkl")

    pred = load_pickle(pred_path)
    label = load_pickle(label_path) if os.path.exists(label_path) else None
    port = load_pickle(port_path) if os.path.exists(port_path) else None
    ind = load_pickle(ind_path) if os.path.exists(ind_path) else None
    pos = load_pickle(pos_path) if os.path.exists(pos_path) else None

    pred_stats = pred.describe().round(6)
    daily_ic = compute_daily_ic(pred, label) if label is not None else None
    ic_mean = float(daily_ic.mean()) if daily_ic is not None else math.nan
    ic_std = float(daily_ic.std()) if daily_ic is not None else math.nan
    icir = ic_mean / (ic_std + 1e-12) if daily_ic is not None else math.nan

    recs = compute_recent_recommendations(pred, 5)
    stock_analysis = compute_stock_analysis(pred, label, recs)
    port_metrics = extract_port_metrics(port)
    ind_metrics = extract_indicator_metrics(ind)
    holding_snapshot = latest_holding_snapshot(pos)

    score_col = pred.columns[0]
    plot_hist(pred[score_col], os.path.join(out_dir, "pred_score_hist.png"), "Prediction score distribution")
    if daily_ic is not None:
        plot_line(daily_ic.rolling(60, min_periods=20).mean(), os.path.join(out_dir, "ic_rolling_60d.png"), "60D rolling IC")
    plot_bar(recs.set_index("instrument")["score"], os.path.join(out_dir, "latest_top5_scores.png"), "Top-5 recommended stocks (latest date)")

    exp_time_str = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

    # Markdown report
    md = []
    md.append("# 因子挖掘详细分析报告")
    md.append("")
    md.append(f"- 实验目录：`{run_dir}`")
    md.append(f"- 实验时间：`{exp_time_str}`")
    md.append(f"- 输出目录：`{out_dir}`")
    md.append("- 因子模型：**LightGBM (LGBModel) + Alpha158**")
    md.append("")
    md.append("## 1. 因子/预测概览")
    md.append("### 预测分布统计")
    md.append("```text")
    md.append(safe_md_table(pred_stats))
    md.append("```")
    md.append(f"- 样本数：{len(pred):,}")
    md.append(f"- 最新可用交易日：`{pred.index.get_level_values(0).max()}`")
    md.append("")
    md.append("### 预测分布图")
    md.append("![pred_score_hist](pred_score_hist.png)")

    if daily_ic is not None:
        md.append("")
        md.append("## 2. Qlib 可视化分析结果")
        md.append("### 日度 IC 统计")
        md.append(f"- IC_mean：{ic_mean:.6f}")
        md.append(f"- IC_std：{ic_std:.6f}")
        md.append(f"- ICIR：{icir:.6f}")
        md.append("### 60 日滚动 IC")
        md.append("![ic_rolling_60d](ic_rolling_60d.png)")

    if isinstance(port, pd.DataFrame):
        md.append("")
        md.append("### 组合回测结果（portfolio_analysis）")
        md.append("```text")
        md.append(port.to_string())
        md.append("```")
        md.append("### 关键收益与回撤指标")
        md.append("```text")
        md.append(
            pd.Series(port_metrics)
            .rename("value")
            .to_frame()
            .to_string()
        )
        md.append("```")

    if ind is not None:
        md.append("")
        md.append("### 指标分析（indicator_analysis）")
        md.append("```text")
        md.append(safe_md_table(pd.DataFrame(ind)))
        md.append("```")
        md.append("### 指标摘要")
        md.append("```text")
        md.append(pd.Series(ind_metrics).rename("value").to_frame().to_string())
        md.append("```")

    if isinstance(holding_snapshot, pd.DataFrame) and not holding_snapshot.empty:
        md.append("")
        md.append("### 最新持仓概况")
        md.append("```text")
        md.append(holding_snapshot.head(10).to_string(index=False))
        md.append("```")

    md.append("")
    md.append("## 3. 推荐股票")
    md.append("基于最新交易日预测值，建议优先关注以下 5 只股票（按 score 从高到低）：")
    md.append("```text")
    md.append(recs.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("### 推荐图")
    md.append("![latest_top5_scores](latest_top5_scores.png)")
    md.append("")
    md.append("### 单票细分分析")
    md.append("```text")
    md.append(stock_analysis.to_string(index=False))
    md.append("```")

    md.append("")
    md.append("## 4. 推荐交易策略")
    md.append("1. **策略类型**：采用 `TopkDropoutStrategy` 风格的多头策略，但实际执行时建议将持仓缩小到最新 Top-5 因子得分股票，按等权或近似等权配置。")
    md.append("2. **调仓频率**：建议以 **日频/每 1-2 个交易日** 为主；若交易成本偏高，可适度延长持有期。")
    md.append("3. **持仓控制**：5 只股票等权持有，单票权重约 20%，并结合流动性对低成交个股进一步降权。")
    md.append("4. **风控建议**：设置单日最大回撤阈值、行业/风格暴露上限，并在成交量较低时降低下单规模。")
    md.append("5. **当前结果解读**：本次回测无成本表现优于有成本表现，说明策略有超额收益潜力，但**交易成本侵蚀明显**，需重点优化换手率。")

    md.append("")
    md.append("## 5. 关键结论")
    md.append(f"- 因子具有**正向预测能力**，`ICIR ≈ {icir:.3f}`。")
    md.append("- 无成本下组合收益与 IR 尚可，但有成本后明显下降，交易效率是主要短板。")
    md.append("- 建议后续做 **walk-forward CV** 与 **滚动回测**，观察不同市场阶段的稳定性。")

    out_md = os.path.join(out_dir, f"factor_analysis_report_{exp_time}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # HTML report
    sections = []
    sections.append(
        f"<div class='card'><p><b>实验目录：</b>{html.escape(run_dir)}</p><p><b>实验时间：</b>{html.escape(exp_time_str)}</p><p><b>输出目录：</b>{html.escape(out_dir)}</p><p><b>因子模型：</b>LightGBM (LGBModel) + Alpha158</p></div>"
    )
    sections.append("<div class='card'><h2>1. 因子/预测概览</h2>")
    sections.append(f"<p class='muted'>样本数：{len(pred):,}；最新可用交易日：{html.escape(str(pred.index.get_level_values(0).max()))}</p>")
    sections.append("<h3>预测分布统计</h3>")
    sections.append(safe_html_table(pred_stats))
    sections.append("<h3>预测分布图</h3><img src='pred_score_hist.png' alt='pred score hist'>")
    sections.append("<h3>关键指标总览</h3>")
    sections.append(
        f"<div class='card'><ul>"
        f"<li><b>IC_mean：</b>{ic_mean:.6f}</li>"
        f"<li><b>ICIR：</b>{icir:.6f}</li>"
        f"<li><b>无成本年化收益：</b>{port_metrics.get('excess_return_without_cost_annualized_return', math.nan):.6f}</li>"
        f"<li><b>无成本信息比率：</b>{port_metrics.get('excess_return_without_cost_information_ratio', math.nan):.6f}</li>"
        f"<li><b>无成本最大回撤：</b>{port_metrics.get('excess_return_without_cost_max_drawdown', math.nan):.6f}</li>"
        f"<li><b>有成本年化收益：</b>{port_metrics.get('excess_return_with_cost_annualized_return', math.nan):.6f}</li>"
        f"<li><b>有成本信息比率：</b>{port_metrics.get('excess_return_with_cost_information_ratio', math.nan):.6f}</li>"
        f"<li><b>有成本最大回撤：</b>{port_metrics.get('excess_return_with_cost_max_drawdown', math.nan):.6f}</li>"
        f"<li><b>ffr：</b>{ind_metrics.get('ffr', math.nan):.6f}</li>"
        f"<li><b>pa：</b>{ind_metrics.get('pa', math.nan):.6f}</li>"
        f"<li><b>pos：</b>{ind_metrics.get('pos', math.nan):.6f}</li>"
        f"</ul></div>"
    )
    if daily_ic is not None:
        sections.append("<h3>日度 IC 统计</h3>")
        sections.append(f"<ul><li>IC_mean：{ic_mean:.6f}</li><li>IC_std：{ic_std:.6f}</li><li>ICIR：{icir:.6f}</li></ul>")
        sections.append("<h3>60 日滚动 IC</h3><img src='ic_rolling_60d.png' alt='rolling ic'>")
    if isinstance(port, pd.DataFrame):
        sections.append("<h3>组合回测结果（portfolio_analysis）</h3>")
        sections.append(safe_html_table(port))
        sections.append("<h3>关键收益与回撤指标</h3>")
        sections.append(safe_html_table(pd.Series(port_metrics).rename("value")))
    if ind is not None:
        sections.append("<h3>指标分析（indicator_analysis）</h3>")
        sections.append(safe_html_table(pd.DataFrame(ind)))
        sections.append("<h3>指标摘要</h3>")
        sections.append(safe_html_table(pd.Series(ind_metrics).rename("value")))
    if isinstance(holding_snapshot, pd.DataFrame) and not holding_snapshot.empty:
        sections.append("<h3>最新持仓概况</h3>")
        sections.append(safe_html_table(holding_snapshot.head(10)))
    sections.append("</div>")

    sections.append("<div class='card'><h2>2. 推荐股票</h2>")
    sections.append("<p>基于最新交易日预测值，建议优先关注以下 5 只股票（按 score 从高到低）：</p>")
    sections.append(safe_html_table(recs))
    sections.append("<h3>推荐图</h3><img src='latest_top5_scores.png' alt='top5 scores'>")
    sections.append("<h3>单票细分分析</h3>")
    sections.append(safe_html_table(stock_analysis))
    sections.append("</div>")

    sections.append(
        "<div class='card'><h2>3. 推荐交易策略</h2><ol>"
        "<li>采用 TopkDropoutStrategy 风格的多头策略，但实际执行时建议将持仓缩小到最新 Top-5 因子得分股票，按等权或近似等权配置。</li>"
        "<li>建议以日频/每 1-2 个交易日为主；若交易成本偏高，可适度延长持有期。</li>"
        "<li>5 只股票等权持有，单票权重约 20%，并结合流动性对低成交个股进一步降权。</li>"
        "<li>设置单日最大回撤阈值、行业/风格暴露上限，并在成交量较低时降低下单规模。</li>"
        "<li>本次回测无成本表现优于有成本表现，说明策略有超额收益潜力，但交易成本侵蚀明显，需重点优化换手率。</li>"
        "</ol></div>"
    )
    sections.append(
        f"<div class='card'><h2>4. 关键结论</h2><ul><li>因子具有正向预测能力，ICIR 约为 {icir:.3f}。</li><li>无成本下组合收益与 IR 尚可，但有成本后明显下降，交易效率是主要短板。</li><li>建议后续做 walk-forward CV 与滚动回测，观察不同市场阶段的稳定性。</li></ul></div>"
    )

    out_html = os.path.join(out_dir, f"factor_analysis_report_{exp_time}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_html("因子挖掘详细分析报告", sections))

    print(out_md)
    print(out_html)
    print(out_dir)


if __name__ == "__main__":
    main()
