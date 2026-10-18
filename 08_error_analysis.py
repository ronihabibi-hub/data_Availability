"""
08_error_analysis.py (version 3)
Where do the out-of-fold errors come from?  Uses the generator metadata
(benign application profile / flood variant / stealth flag; never used as a
model feature) and the out-of-fold predictions from 02_nested_cv_evaluation.py.
Output: error_analysis.json
"""
import json, os, importlib, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
md = pd.read_csv(os.path.join(HERE, 'network_flood_dataset_metadata.csv')); df = pd.read_csv(os.path.join(HERE, 'network_flood_dataset.csv'))
oof = pd.read_csv(os.path.join(HERE, 'oof_predictions.csv'))
y = df.Label.values; order = np.concatenate([te for _, te in StratifiedKFold(5, shuffle=True, random_state=77).split(df.drop(columns='Label').values, y)])
assert (y[order] == oof.label.values).all()
m = md.iloc[order].reset_index(drop=True); out = {}
for model in ['KNN', 'DecisionTree', 'RandomForest']:
    wrong = (oof[model + '_pred'] != oof.label).values; r = {}
    for prof, g in m.groupby('Profile'):
        r[prof] = dict(n=int(len(g)), errors=int(wrong[g.index].sum()), error_rate_pct=float(wrong[g.index].mean() * 100))
    fl = m[m.Label == 1]
    for st in [0, 1]:
        g = fl[fl.Stealth == st]; r[f'flood_stealth_{st}'] = dict(n=int(len(g)), errors=int(wrong[g.index].sum()), error_rate_pct=float(wrong[g.index].mean() * 100))
    out[model] = r
json.dump(out, open(os.path.join(HERE, 'error_analysis.json'), 'w'), indent=2)
for model, r in out.items():
    print(model, {k: (v['errors'], v['n']) for k, v in r.items()})
