#!/usr/bin/env python3
"""Analyze MLflow run artifacts and produce a markdown report with plots.

Usage:
  python scripts/analyze_mlrun_run.py --run mlruns/860985181741339330/86667b9d122e4f479968aebd9c8892ea --out DATA/analysis_outputs/rdagent_one_run_analysis
"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def annualized_return(rts, periods_per_year=252):
    mean = np.nanmean(rts)
    return (1 + mean) ** periods_per_year - 1


def sharpe(rts, periods_per_year=252):
    mu = np.nanmean(rts) * periods_per_year
    sigma = np.nanstd(rts) * np.sqrt(periods_per_year)
    return mu / (sigma + 1e-9)


def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-9)
    return drawdown.min()


def plot_series(series, out_path, title=""):
    plt.figure(figsize=(10, 4))
    series.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def df_to_md(obj):
    """Render DataFrame/Series-like objects to markdown without requiring tabulate."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name or "value")
    if hasattr(obj, "to_string"):
        return "```text\n" + obj.to_string() + "\n```"
    return f"```text\n{obj}\n```"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run_dir = args.run
    out_dir = args.out
    ensure_dir(out_dir)

    artifacts_dir = os.path.join(run_dir, "artifacts")

    # Attempt to find common artifacts
    pred_path = os.path.join(artifacts_dir, "pred.pkl")
    port_path = os.path.join(artifacts_dir, "portfolio_analysis", "port_analysis_1day.pkl")
    ind_path = os.path.join(artifacts_dir, "portfolio_analysis", "indicator_analysis_1day.pkl")

    report_lines = []
    report_lines.append(f"# Analysis Report for {run_dir}\n")

    if os.path.exists(pred_path):
        pred = load_pickle(pred_path)
        report_lines.append("## Predictions (sample)\n")
        report_lines.append(df_to_md(pred.head()))
        # compute IC if label exists in mlruns artifacts (label.pkl)
        # but we will compute simple stats of predictions
        report_lines.append("\n## Prediction stats\n")
        report_lines.append(df_to_md(pred.describe()))
        # plot mean score over time if datetime index
        try:
            if isinstance(pred.index, pd.DatetimeIndex):
                s = pred.groupby(level=0).mean().iloc[:, 0]
                plot_series(s.cumsum(), os.path.join(out_dir, "pred_cumsum.png"), "Cumulative mean prediction")
                report_lines.append("\n![pred_cumsum](pred_cumsum.png)\n")
        except Exception:
            pass
    else:
        report_lines.append("pred.pkl not found.\n")

    if os.path.exists(port_path):
        port = load_pickle(port_path)
        report_lines.append("## Portfolio Analysis (1day)\n")
        # port expected to be a DataFrame or dict with 'risk' table
        try:
            if isinstance(port, pd.DataFrame):
                report_lines.append(df_to_md(port))
            elif isinstance(port, dict) and "risk" in port:
                risk = port["risk"]
                report_lines.append(df_to_md(risk))
            elif hasattr(port, "to_frame"):
                report_lines.append(df_to_md(pd.DataFrame(port)))
        except Exception:
            report_lines.append(str(port))
        # try to plot cumulative returns if present
        try:
            if isinstance(port, pd.DataFrame) and "risk" in port.columns:
                pass
            elif isinstance(port, dict) and "returns" in port:
                r = pd.Series(port["returns"])
                cr = (1 + r).cumprod()
                plot_series(cr, os.path.join(out_dir, "portfolio_cumret.png"), "Portfolio cumulative returns")
                report_lines.append("\n![portfolio_cumret](portfolio_cumret.png)\n")
        except Exception:
            pass
    else:
        report_lines.append("port_analysis_1day.pkl not found.\n")

    if os.path.exists(ind_path):
        ind = load_pickle(ind_path)
        report_lines.append("## Indicator Analysis (1day)\n")
        try:
            report_lines.append(df_to_md(pd.DataFrame(ind)))
        except Exception:
            report_lines.append(str(ind))
    else:
        report_lines.append("indicator_analysis_1day.pkl not found.\n")

    # Basic aggregated metrics if pred + labels available
    label_path = os.path.join(artifacts_dir, "label.pkl")
    if os.path.exists(pred_path) and os.path.exists(label_path):
        lab = load_pickle(label_path)
        pred = load_pickle(pred_path)
        try:
            # compute IC per date
            df = pd.concat([pred, lab], axis=1, join='inner')
            df.columns = ["pred", "label"]
            # group by date (if multiindex)
            if isinstance(df.index, pd.MultiIndex):
                dates = df.index.get_level_values(0)
                df2 = df.copy()
                df2.index = dates
                ic = df2.groupby(df2.index).apply(lambda g: g['pred'].corr(g['label']))
            else:
                ic = pd.Series([df['pred'].corr(df['label'])])
            report_lines.append("\n## IC / ICIR\n")
            report_lines.append(str({'IC_mean': ic.mean(), 'IC_std': ic.std(), 'ICIR': ic.mean() / (ic.std() + 1e-9)}))
        except Exception as e:
            report_lines.append(f"Failed to compute IC: {e}\n")

    # write report
    out_report = os.path.join(out_dir, "analysis_report.md")
    with open(out_report, "w") as f:
        f.write("\n\n".join(report_lines))

    print("Report written to:", out_report)


if __name__ == "__main__":
    main()
