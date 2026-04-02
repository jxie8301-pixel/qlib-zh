#!/usr/bin/env python3
"""
Find best strategy in backtest summary and export positions to mlflow artifacts.

Usage:
  python3 scripts/analysis/export_best_positions.py
"""
from pathlib import Path
import pickle
import sys
import pandas as pd

import importlib.util
# load backtest_param_grid module by path to avoid package import issues
mod_path = Path(__file__).resolve().parent / 'backtest_param_grid.py'
spec = importlib.util.spec_from_file_location('backtest_param_grid', str(mod_path))
bt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt_mod)
to_series = bt_mod.to_series
ensure_midx = bt_mod.ensure_midx
run_backtest = bt_mod.run_backtest

ROOT = Path('/Users/apple/github/qlib/DATA/analysis_outputs/2026-3-31')
ART = ROOT / 'mlflow_run' / 'artifacts'
SUMMARY = ROOT / 'backtest_grid_summary.csv'


def parse_strategy_row(row):
    # row contains columns: hold, weighting, cost, slippage, turnover_limit
    return dict(
        hold=int(row['hold']),
        weighting=row['weighting'],
        cost=float(row.get('cost', 0.0)),
        slippage=float(row.get('slippage', 0.0)),
        turnover_limit=float(row.get('turnover_limit', 1.0)),
    )


def main():
    if not SUMMARY.exists():
        print('Summary CSV not found:', SUMMARY)
        return
    df = pd.read_csv(SUMMARY)
    if df.empty:
        print('Empty summary')
        return
    # choose best by sharpe
    best = df.sort_values('sharpe', ascending=False).iloc[0]
    params = parse_strategy_row(best)
    print('Best strategy chosen:', best['strategy'], 'params:', params)

    # load pred & label and build joined df
    pred_p = ART / 'pred.pkl'
    label_p = ART / 'label.pkl'
    if not pred_p.exists() or not label_p.exists():
        print('pred.pkl or label.pkl missing in artifacts')
        return
    pred = to_series(pickle.load(open(pred_p, 'rb')))
    label = to_series(pickle.load(open(label_p, 'rb')))
    pred = ensure_midx(pred).rename('score')
    label = ensure_midx(label).rename('label')
    joined = pred.to_frame().join(label.to_frame(), how='inner')

    drs, metrics, positions = run_backtest(joined, top_q=0.1, hold=params['hold'], weighting=params['weighting'], cost=params['cost'], slippage=params['slippage'], turnover_limit=params['turnover_limit'], return_positions=True)

    out_dir = ART / 'portfolio_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'positions_best.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(positions, f)

    print('Saved best positions to', out_path)


if __name__ == '__main__':
    main()
