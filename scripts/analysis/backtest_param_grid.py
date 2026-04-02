#!/usr/bin/env python3
"""
Parametric backtest grid: vary holding period, weighting and transaction cost.

Generates:
 - DATA/analysis_outputs/2026-3-31/backtest_grid_summary.csv
 - DATA/analysis_outputs/2026-3-31/backtest_grid_cumrets.png

Usage:
  python3 scripts/analysis/backtest_param_grid.py
"""
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path('/Users/apple/github/qlib/DATA/analysis_outputs/2026-3-31')
ART = ROOT / 'mlflow_run' / 'artifacts'
OUT_CSV = ROOT / 'backtest_grid_summary.csv'
OUT_PNG = ROOT / 'backtest_grid_cumrets.png'


def load_pickle(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def to_series(obj):
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if 'score' in obj.columns:
            return obj['score']
        return obj.iloc[:, 0]
    return pd.Series(obj)


def ensure_midx(s):
    # Ensure Series has MultiIndex (date, instrument)
    idx = s.index
    if isinstance(idx, pd.MultiIndex) and idx.nlevels >= 2:
        dates = pd.to_datetime(idx.get_level_values(0))
        inst = idx.get_level_values(1)
        s.index = pd.MultiIndex.from_arrays([dates, inst])
        return s
    # try parse single-level with tuple-like
    try:
        new_index = []
        for i in s.index:
            if isinstance(i, tuple) and len(i) >= 2:
                new_index.append((pd.to_datetime(i[0]), i[1]))
            else:
                new_index.append((pd.to_datetime(i), None))
        s.index = pd.MultiIndex.from_tuples(new_index)
    except Exception:
        pass
    return s


def compute_metrics(dr):
    dr = dr.dropna()
    if dr.empty:
        return dict(total=0, ann_ret=0, ann_vol=0, sharpe=0, mdd=0)
    cum = (1 + dr).cumprod() - 1
    total = cum.iloc[-1]
    ann = (1 + total) ** (252.0 / len(dr)) - 1
    vol = dr.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    peak = (1 + dr).cumprod().cummax()
    dd = (1 + dr).cumprod() / peak - 1
    mdd = dd.min()
    return dict(total=total, ann_ret=ann, ann_vol=vol, sharpe=sharpe, mdd=mdd)


def run_backtest(df, top_q=0.1, hold=1, weighting='equal', cost=0.0, slippage=0.0, turnover_limit=1.0, return_positions=False):
    """
    Run backtest with optional slippage and daily turnover limit.

    - `cost` is per-unit transaction cost (commission) applied to turnover.
    - `slippage` is per-unit price impact applied to turnover (e.g., 0.0005 = 5bp).
    - `turnover_limit` caps daily turnover (fraction). If raw turnover exceeds the limit,
      the weight changes are scaled down proportionally so applied turnover == turnover_limit.
    """

    df = df.dropna(subset=['score'])
    dates = sorted(set([d for d, _ in df.index]))

    positions_by_date = {}
    # compute target weights for each formation date
    targets = {}
    for date in dates:
        day_idx = df.index.get_level_values(0) == date
        day_df = df[day_idx]
        if day_df.empty:
            targets[date] = {}
            continue
        cutoff = day_df['score'].quantile(1 - top_q)
        sel = day_df[day_df['score'] >= cutoff]
        if sel.empty:
            targets[date] = {}
            continue
        if weighting == 'equal':
            w = np.repeat(1.0, len(sel))
        else:
            scores = sel['score'].values
            minv = scores.min()
            if minv < 0:
                scores = scores - minv
            w = scores.clip(min=0.0)
            if w.sum() == 0:
                w = np.repeat(1.0, len(w))
        w = w / w.sum()
        targets[date] = dict(zip(sel.index.get_level_values(1), w))

    # apply targets for hold days
    for i, date in enumerate(dates):
        tgt = targets.get(date, {})
        for k in range(hold):
            if i + k >= len(dates):
                break
            dapply = dates[i + k]
            positions_by_date.setdefault(dapply, {})
            positions_by_date[dapply] = tgt

    # simulate day by day with turnover limit and slippage
    daily_returns = []
    prev_weights = {}
    for date in dates:
        targ = positions_by_date.get(date, {})

        # compute raw turnover
        instruments = set(targ.keys()) | set(prev_weights.keys())
        raw_turnover = sum(abs(targ.get(i, 0.0) - prev_weights.get(i, 0.0)) for i in instruments)

        # apply turnover cap
        if raw_turnover > turnover_limit:
            scale = turnover_limit / raw_turnover if raw_turnover > 0 else 0.0
            # scaled applied weights = prev + (targ - prev) * scale
            applied = {}
            for inst in instruments:
                applied[inst] = prev_weights.get(inst, 0.0) + (targ.get(inst, 0.0) - prev_weights.get(inst, 0.0)) * scale
            applied_turnover = sum(abs(applied.get(i, 0.0) - prev_weights.get(i, 0.0)) for i in instruments)
        else:
            applied = targ
            applied_turnover = raw_turnover

        # compute portfolio return using applied weights
        mask = df.index.get_level_values(0) == date
        day_df = df[mask]
        ret = np.nan
        if applied and not day_df.empty:
            # build quick map inst -> label for this date
            insts = day_df.index.get_level_values(1)
            labels_vals = day_df['label'].values
            label_map = {inst: val for inst, val in zip(insts, labels_vals)}
            vals = []
            for inst, w in applied.items():
                r = label_map.get(inst, np.nan)
                if not pd.isna(r):
                    vals.append((w, r))
            if vals:
                total_w = sum(w for w, _ in vals)
                if total_w == 0:
                    ret = np.nan
                else:
                    ret = sum(w * r for w, r in vals) / total_w

        # cost: commission + slippage applied to applied_turnover
        cost_amount = applied_turnover * cost + applied_turnover * slippage
        if pd.notna(ret):
            ret = ret - cost_amount

        daily_returns.append((date, ret))
        prev_weights = {k: v for k, v in applied.items() if v != 0.0}

    drs = pd.Series([r for _, r in daily_returns], index=[d for d, _ in daily_returns])
    metrics = compute_metrics(drs)
    if return_positions:
        return drs, metrics, positions_by_date
    return drs, metrics


def main():
    pred_p = ART / 'pred.pkl'
    label_p = ART / 'label.pkl'
    if not pred_p.exists() or not label_p.exists():
        print('pred.pkl or label.pkl not found in', ART)
        sys.exit(1)

    pred = to_series(load_pickle(pred_p))
    label = to_series(load_pickle(label_p))

    # build joined DataFrame once to speed repeated backtests
    pred = ensure_midx(pred).rename('score')
    label = ensure_midx(label).rename('label')
    df = pred.to_frame().join(label.to_frame(), how='inner')

    hold_list = [1, 5, 10]
    weightings = ['equal', 'score']
    costs = [0.0, 0.001, 0.002]
    slippages = [0.0, 0.0005, 0.001]
    turnover_limits = [1.0, 0.2, 0.1]

    rows = []
    cumret_series = {}

    for hold in hold_list:
        for w in weightings:
            for c in costs:
                for s in slippages:
                    for tlimit in turnover_limits:
                        name = f'h{hold}_{w}_c{c}_s{s}_t{tlimit}'
                        print('Running', name)
                        drs, m = run_backtest(df, top_q=0.1, hold=hold, weighting=w, cost=c, slippage=s, turnover_limit=tlimit)
                        rows.append(dict(strategy=name, hold=hold, weighting=w, cost=c, slippage=s, turnover_limit=tlimit, total=m.get('total', 0), ann_ret=m.get('ann_ret', 0), ann_vol=m.get('ann_vol', 0), sharpe=m.get('sharpe', 0), mdd=m.get('mdd', 0)))
                        cumret_series[name] = (1 + drs.fillna(0)).cumprod() - 1

    dfres = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
    dfres.to_csv(OUT_CSV, index=False)

    plt.figure(figsize=(10, 6))
    for k, s in cumret_series.items():
        plt.plot(s.index, s.values, label=k)
    plt.legend()
    plt.grid(True)
    plt.title('Grid cumulative returns')
    plt.tight_layout()
    plt.savefig(OUT_PNG)

    print('Saved summary to', OUT_CSV, 'and plot to', OUT_PNG)


if __name__ == '__main__':
    main()
