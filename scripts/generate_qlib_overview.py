#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio


REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import qlib  # noqa: F401
except Exception:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_position


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_signal_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame(col_name)
    df = df.copy()
    df = df.iloc[:, :1]
    df.columns = [col_name]
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if len(names) >= 2 and names[0] == "datetime" and names[1] == "instrument":
            df = df.swaplevel().sort_index()
        elif len(names) >= 2 and names[0] != "instrument" and names[1] == "datetime":
            df.index = df.index.set_names(["instrument", "datetime"])
    return df.sort_index()


def normalize_report_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame):
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
    return out.sort_index()


def build_analysis_df(report_df: pd.DataFrame, port_df: pd.DataFrame | None) -> pd.DataFrame:
    if isinstance(port_df, pd.DataFrame) and not port_df.empty:
        return port_df
    analysis = {
        "excess_return_without_cost": risk_analysis(report_df["return"] - report_df["bench"]),
        "excess_return_with_cost": risk_analysis(report_df["return"] - report_df["bench"] - report_df["cost"]),
    }
    return pd.concat(analysis)


def safe_figures(builder, *args, **kwargs):
    try:
        figures = builder(*args, **kwargs)
        return list(figures or []), None
    except Exception as exc:
        return [], str(exc)


def figure_block(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def section_block(title: str, figures: list, error: str | None = None, desc: str | None = None) -> str:
    parts = ["<section class='card'>", f"<h2>{html.escape(title)}</h2>"]
    if desc:
        parts.append(f"<p class='muted'>{html.escape(desc)}</p>")
    if error:
        parts.append(f"<p class='warn'>生成失败：{html.escape(error)}</p>")
    elif not figures:
        parts.append("<p class='muted'>无可用图表。</p>")
    else:
        for fig in figures:
            parts.append("<div class='figure'>")
            parts.append(figure_block(fig))
            parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def render_overview(title: str, summary_rows: list[tuple[str, str]], sections: list[str]) -> str:
    summary_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in summary_rows if v is not None
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{html.escape(title)}</title>
  <script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif; margin: 24px auto; max-width: 1380px; color: #222; padding: 0 16px; background: #fafafa; }}
    h1, h2, h3 {{ line-height: 1.2; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 20px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    .muted {{ color: #6b7280; }}
    .warn {{ color: #b45309; background: #fff7ed; border: 1px solid #fed7aa; padding: 10px 12px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 220px; background: #f9fafb; }}
    .figure {{ margin: 16px 0 24px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <section class='card'>
    <h2>实验概览</h2>
    <table>{summary_html}</table>
  </section>
  {''.join(sections)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qlib native visualization overview.html from an MLflow run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing meta.yaml and artifacts/")
    parser.add_argument("--output", required=True, help="Output HTML path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts_dir = run_dir / "artifacts"
    pred = normalize_signal_df(load_pickle(artifacts_dir / "pred.pkl"), "score")
    label = normalize_signal_df(load_pickle(artifacts_dir / "label.pkl"), "label")
    pred_label = pd.concat([pred, label], axis=1, join="inner").dropna().sort_index()

    report_df = normalize_report_df(load_pickle(artifacts_dir / "portfolio_analysis" / "report_normal_1day.pkl"))
    port_df = load_pickle(artifacts_dir / "portfolio_analysis" / "port_analysis_1day.pkl")
    positions = load_pickle(artifacts_dir / "portfolio_analysis" / "positions_normal_1day.pkl")

    analysis_df = build_analysis_df(report_df, port_df)
    start_date = str(report_df.index.min().date()) if isinstance(report_df.index, pd.DatetimeIndex) else None
    end_date = str(report_df.index.max().date()) if isinstance(report_df.index, pd.DatetimeIndex) else None

    sections = []

    report_figs, report_err = safe_figures(analysis_position.report_graph, report_df, show_notebook=False)
    sections.append(section_block("组合回测报告", report_figs, report_err, "qlib.analysis_position.report_graph"))

    score_ic_figs, score_ic_err = safe_figures(analysis_position.score_ic_graph, pred_label, show_notebook=False)
    sections.append(section_block("Score IC 分析", score_ic_figs, score_ic_err, "qlib.analysis_position.score_ic_graph"))

    risk_figs, risk_err = safe_figures(
        analysis_position.risk_analysis_graph,
        analysis_df=analysis_df,
        report_normal_df=report_df,
        show_notebook=False,
    )
    sections.append(section_block("风险分析", risk_figs, risk_err, "qlib.analysis_position.risk_analysis_graph"))

    cumret_figs, cumret_err = safe_figures(
        analysis_position.cumulative_return_graph,
        positions,
        report_df,
        label,
        start_date=start_date,
        end_date=end_date,
        show_notebook=False,
    )
    sections.append(section_block("买卖持有累计收益", cumret_figs, cumret_err, "qlib.analysis_position.cumulative_return_graph"))

    rank_figs, rank_err = safe_figures(
        analysis_position.rank_label_graph,
        positions,
        label,
        start_date=start_date,
        end_date=end_date,
        show_notebook=False,
    )
    sections.append(section_block("持仓标签分位", rank_figs, rank_err, "qlib.analysis_position.rank_label_graph"))

    summary_rows = [
        ("run_dir", str(run_dir)),
        ("artifact_dir", str(artifacts_dir)),
        ("prediction_rows", f"{len(pred):,}"),
        ("pred_label_rows", f"{len(pred_label):,}"),
        ("start_date", start_date or ""),
        ("end_date", end_date or ""),
        ("report_columns", ", ".join(map(str, report_df.columns.tolist()))),
    ]

    html_doc = render_overview("Qlib 可视化分析总览", summary_rows, sections)
    output_path.write_text(html_doc, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()