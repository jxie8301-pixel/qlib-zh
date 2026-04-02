#!/usr/bin/env python3
"""
Generate factor selection and simple backtest report for 2026-3-31 outputs.

Usage:
  python scripts/analysis/generate_report_2026_3_31.py

This script expects artifacts under the run copied directory:
  /Users/apple/github/qlib/DATA/analysis_outputs/2026-3-31/mlflow_run/artifacts

It loads `pred.pkl` and `label.pkl` (pickled pandas objects), computes
top-quantile selection returns and writes a markdown report and a plot
into the analysis output folder.
"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path('/Users/apple/github/qlib/DATA/analysis_outputs/2026-3-31')
ARTIFACTS = ROOT / 'mlflow_run' / 'artifacts'
OUT_MD = ROOT / 'factor_report.md'
OUT_PNG = ROOT / 'factor_cumret.png'


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_series(pred):
    # Normalize pred into a Series indexed by (datetime, instrument)
    if isinstance(pred, pd.Series):
        return pred
    if isinstance(pred, pd.DataFrame):
        # prefer column named 'score' or first column
        if 'score' in pred.columns:
            return pred['score']
        return pred.iloc[:, 0]
    # try numpy array with index attribute
    try:
        return pd.Series(pred)
    except Exception:
        raise ValueError('Unsupported pred object type: %s' % type(pred))


def compute_metrics(daily_ret):
    # daily_ret: pd.Series indexed by date
    dr = daily_ret.dropna()
    if dr.empty:
        return {}
    cum = (1 + dr).cumprod() - 1
    total_ret = cum.iloc[-1]
    ann_ret = (1 + total_ret) ** (252.0 / len(dr)) - 1
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    # max drawdown
    peak = (1 + dr).cumprod().cummax()
    dd = (1 + dr).cumprod() / peak - 1
    mdd = dd.min()
    return dict(total=total_ret, ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd)


def main():
    if not ARTIFACTS.exists():
        print('Artifacts folder not found:', ARTIFACTS, file=sys.stderr)
        sys.exit(1)

    pred_path = ARTIFACTS / 'pred.pkl'
    label_path = ARTIFACTS / 'label.pkl'

    if not pred_path.exists():
        print('pred.pkl not found in artifacts', file=sys.stderr)
        sys.exit(1)

    pred = load_pickle(pred_path)
    pred_s = to_series(pred)

    # pred index may be MultiIndex (datetime, instrument) or other
    # Try to convert index first level to datetime
    idx = pred_s.index
    if isinstance(idx, pd.MultiIndex) and idx.nlevels >= 2:
        dates = pd.to_datetime(idx.get_level_values(0))
        inst = idx.get_level_values(1)
        pred_s.index = pd.MultiIndex.from_arrays([dates, inst])
    else:
        # try to parse single-level date index
        try:
            pd.to_datetime(pred_s.index)
            pred_s.index = pd.to_datetime(pred_s.index)
        except Exception:
            pass

    # load label (forward return) if available
    label = None
    if label_path.exists():
        label = load_pickle(label_path)

    # Align labels if possible
    if label is not None:
        try:
            if isinstance(label, pd.Series):
                labels = label
            elif isinstance(label, pd.DataFrame):
                labels = label.iloc[:, 0]
            else:
                labels = pd.Series(label)
        except Exception:
            labels = None
    else:
        labels = None

    # prepare per-date selection returns
    quantiles = [0.05, 0.1, 0.2]
    results = {}

    # If labels available and share index with pred_s
    if labels is not None:
        # Ensure labels have MultiIndex aligned to pred
        # Convert labels similar to pred
        if isinstance(labels.index, pd.MultiIndex) and isinstance(pred_s.index, pd.MultiIndex):
            pass
        # compute daily portfolio returns for each quantile
        pred_df = pred_s.rename('score').to_frame()
        pred_df['date'] = [i[0] if isinstance(i, tuple) else i for i in pred_df.index]
        # join with labels by index
        merged = pred_df.join(labels.rename('label'), how='inner')
        if 'label' not in merged.columns:
            print('No label column after join; aborting label-based backtest', file=sys.stderr)
            labels = None

    if labels is None:
        # try to use port_analysis positions if present
        pos_path = ARTIFACTS / 'portfolio_analysis' / 'positions_normal_1day.pkl'
        if pos_path.exists():
            pos = load_pickle(pos_path)
            # pos might be dict-like with daily positions; attempt to compute returns if 'value' present
            # Fallback: write message and stop
            print('positions_normal_1day.pkl found but automated parsing not implemented.', file=sys.stderr)
            print('You can provide forward returns (label.pkl) for quantitative backtest.', file=sys.stderr)
            # generate a brief report and exit
            with open(OUT_MD, 'w') as f:
                f.write('# 因子选股与交易策略报告\n\n')
                f.write('自动化回测需要 `label.pkl`（前向收益标签）。\n')
            print('Report written to', OUT_MD)
            return
        else:
            print('No label.pkl and no usable positions found; cannot run backtest', file=sys.stderr)
            with open(OUT_MD, 'w') as f:
                f.write('# 因子选股与交易策略报告\n\n')
                f.write('未找到可用于回测的前向收益标签（label.pkl）或可解析的持仓文件。\n')
            print('Report written to', OUT_MD)
            return

    # Now labels exist and merged available
    merged['date'] = pd.to_datetime(merged['date'])
    grouped = merged.groupby('date')

    daily_ret_df = pd.DataFrame(index=sorted(grouped.groups.keys()))
    for q in quantiles:
        name = f'top_{int(q*100)}pct'
        daily_returns = []
        for date, g in grouped:
            cutoff = g['score'].quantile(1 - q)
            sel = g[g['score'] >= cutoff]
            if sel.empty:
                daily_returns.append(np.nan)
            else:
                # equal-weighted average of label
                daily_returns.append(sel['label'].mean())
        daily_ret_df[name] = pd.Series(daily_returns, index=daily_ret_df.index)

    # compute metrics and cumulative returns plot
    metrics = {name: compute_metrics(daily_ret_df[name].dropna()) for name in daily_ret_df.columns}

    # plot cumulative returns
    plt.figure(figsize=(10, 6))
    for col in daily_ret_df.columns:
        s = daily_ret_df[col].fillna(0)
        plt.plot((1 + s).cumprod() - 1, label=col)
    plt.legend()
    plt.title('Cumulative returns - top quantiles')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG)

    # write markdown report
    with open(OUT_MD, 'w') as f:
        f.write('# 因子选股与交易策略报告\n\n')
        f.write('基于目录: %s\n\n' % ARTIFACTS)
        f.write('## 实验概况\n')
        f.write('- prediction rows: %d\n' % len(pred_s))
        f.write('- date range: %s to %s\n\n' % (merged['date'].min().date(), merged['date'].max().date()))

        f.write('## 策略说明\n')
        f.write('- 每日根据信号 `score` 进行排序，等权构建 top-5%、top-10%、top-20% 组合，持有期 1 日（次日收盘前结算前后）。\n\n')

        f.write('## 回测结果（简要）\n')
        for name, m in metrics.items():
            f.write(f'- **{name}**: total_ret={m.get("total", np.nan):.4f}, ann_ret={m.get("ann_ret", np.nan):.4f}, ann_vol={m.get("ann_vol", np.nan):.4f}, sharpe={m.get("sharpe", np.nan):.2f}, mdd={m.get("mdd", np.nan):.4f}\n')

        f.write('\n![cumret](%s)\n' % OUT_PNG.name)

    print('Report and plot saved to', OUT_MD, OUT_PNG)


if __name__ == '__main__':
    main()
