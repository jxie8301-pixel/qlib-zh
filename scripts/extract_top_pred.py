import pickle
import pandas as pd
from pathlib import Path

p=Path('DATA/analysis_outputs/from_recorder/pred.pkl')
if not p.exists():
    raise SystemExit('pred.pkl not found at ' + str(p))
with p.open('rb') as f:
    pred=pickle.load(f)
print('loaded', type(pred))

if isinstance(pred, pd.DataFrame):
    if isinstance(pred.index, pd.MultiIndex):
        latest = pred.groupby(level=1).last()
        col = latest.columns[0] if len(latest.columns)>0 else None
        if col:
            scores = latest[col].sort_values(ascending=False)
        else:
            scores = latest.iloc[:,0].sort_values(ascending=False)
    else:
        scores = pred.mean(axis=0).sort_values(ascending=False)
else:
    raise SystemExit('pred not a pandas DataFrame; type='+str(type(pred)))

N=30
out = pd.DataFrame({'instrument':scores.head(N).index, 'score':scores.head(N).values})
out_path = Path('DATA/analysis_outputs/from_recorder/topN_pool.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print('Wrote', out_path)
