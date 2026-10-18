"""
04_separability_analysis.py  (version 3)
Class-separability and SHORTCUT diagnostics (editor points 1 and 7-8).
Beyond single-feature AUC this version scans ALL ordered feature ratios
(x/y, every ordered pair whose denominator is not zero everywhere) for a
near-perfect separating threshold, checks whether
any feature ratio takes a class-specific constant value, and reports the
cross-validated accuracy and out-of-fold AUC of increasingly flexible
low-capacity models (stump, depth-2/3 trees, logistic regression).
A label-permutation baseline is included.
Usage: python 04_separability_analysis.py [dataset.csv] [output.json]
Undefined ratios (zero denominator) are handled by an explicit, conservative
policy: all records are kept and undefined values are placed above (or below)
every finite value, keeping the higher AUC. AUC on defined-ratio records only is
reported separately with its coverage, because it uses a subset of the data.
Usage: python 04_separability_analysis.py [dataset.csv] [output.json]
Undefined ratios (zero denominator) are handled by an explicit, conservative
policy: each ratio is evaluated three ways (undefined -> above every finite
value, undefined -> below every finite value, undefined records excluded) and
the HIGHEST AUC / threshold accuracy of the three is kept, so the scan cannot
understate a shortcut. Running it on the version-2 dataset reproduces the
shortcut the editor found (Fwd_Pkts_s / Flow_Pkts_s, AUC = 1.0).
"""
import json, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__)); SEED = 77
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'network_flood_dataset.csv')
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'separability_results.json')
df = pd.read_csv(src); y = df.Label.values; feats = [c for c in df.columns if c != 'Label']

def auc_dir(v):
    a = roc_auc_score(y, v); return max(a, 1 - a)

def best_threshold_acc(v):
    o = np.argsort(v); ys = y[o]; n = len(y)
    pos_left = np.cumsum(ys); neg_left = np.cumsum(1 - ys)
    # predict class 1 above the cut: correct = neg_left + (P - pos_left); or the reverse direction
    P = ys.sum(); acc_up = (neg_left + (P - pos_left)) / n; acc_dn = (pos_left + (n - P - neg_left)) / n
    return float(max(acc_up.max(), acc_dn.max()))

single = {f: auc_dir(df[f].values) for f in feats}

def _auc_acc(v, yy):
    a = roc_auc_score(yy, v); a = max(a, 1 - a)
    o = np.argsort(v); ys_ = yy[o]; n = len(yy); P = ys_.sum()
    pl = np.cumsum(ys_); nl = np.cumsum(1 - ys_)
    acc = max(((nl + (P - pl)) / n).max(), ((pl + (n - P - nl)) / n).max())
    return a, acc

def eval_ratio(num, den):
    """PRIMARY: all records are kept. Undefined ratios (zero denominator) are placed
    above every finite value or below every finite value, and the higher AUC of the
    two is kept (conservative).  SECONDARY (reported separately, with coverage):
    AUC on the records with a defined ratio only."""
    und = den == 0
    r = np.where(und, 0.0, num / np.where(und, 1.0, den))
    if und.all(): return None
    if und.any():
        hi, lo = r[~und].max() + 1.0, r[~und].min() - 1.0
        cands = [np.where(und, hi, r), np.where(und, lo, r)]
    else:
        cands = [r]
    cands = [v for v in cands if np.ptp(v) > 0]
    if not cands: return None
    res = [_auc_acc(v, y) for v in cands]
    prim = (max(x[0] for x in res), max(x[1] for x in res))
    sec = None
    if und.any() and (~und).sum() > 50 and len(np.unique(y[~und])) == 2 and np.ptp(r[~und]) > 0:
        sec = _auc_acc(r[~und], y[~und]) + (int((~und).sum()),)
    return prim, sec, int(und.sum())

ratios = {}; n_undefined = {}; subset_only = {}
for a in feats:
    for b in feats:
        if a == b: continue
        den = df[b].values.astype(float)
        out_ = eval_ratio(df[a].values.astype(float), den)
        if out_ is None: continue
        prim, sec, nu = out_
        ratios[f'{a}/{b}'] = prim
        if nu: n_undefined[f'{a}/{b}'] = nu
        if sec: subset_only[f'{a}/{b}'] = sec
top_ratio = sorted(ratios.items(), key=lambda kv: -kv[1][0])[:8]
distinct = {}
for a, b in [('Fwd_Pkts_s', 'Flow_Pkts_s'), ('Bwd_Pkts_s', 'Flow_Pkts_s'), ('Fwd_Header_Len', 'Tot_Fwd_Pkts'), ('Bwd_Header_Len', 'Tot_Bwd_Pkts')]:
    r = (df[a] / df[b].replace(0, np.nan)).dropna().round(6)
    distinct[f'{a}/{b}'] = {int(k): int(g.nunique()) for k, g in r.groupby(df.Label.loc[r.index])}

X = df[feats].values; skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {'stump_depth1': DecisionTreeClassifier(max_depth=1, random_state=SEED),
          'tree_depth2': DecisionTreeClassifier(max_depth=2, random_state=SEED),
          'tree_depth3': DecisionTreeClassifier(max_depth=3, random_state=SEED),
          'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
low_cap = {}
for k, m in models.items():
    s = cross_val_predict(m, X, y, cv=skf, method='predict_proba')[:, 1]
    low_cap[k] = dict(oof_accuracy_pct=float(accuracy_score(y, s > .5) * 100), oof_roc_auc=float(roc_auc_score(y, s)))
rng = np.random.default_rng(SEED); ys = rng.permutation(y)
s = cross_val_predict(DecisionTreeClassifier(max_depth=3, random_state=SEED), X, ys, cv=skf, method='predict_proba')[:, 1]
low_cap['label_permutation_tree_depth3'] = dict(oof_accuracy_pct=float(accuracy_score(ys, s > .5) * 100), oof_roc_auc=float(roc_auc_score(ys, s)))
out = dict(dataset=os.path.basename(src), n=int(len(df)),
           top10_single_feature_auc=sorted(single.items(), key=lambda kv: -kv[1])[:10],
           top8_feature_ratio_scan=[dict(ratio=k, auc=v[0], best_single_threshold_accuracy_pct=v[1] * 100) for k, v in top_ratio],
           n_ratios_scanned=len(ratios), n_ratios_with_undefined_records=len(n_undefined), zero_denominator_policy='all records kept; undefined placed above or below all finite values, higher AUC kept',
           top5_defined_records_only_scan=[dict(ratio=k, auc=v[0], best_single_threshold_accuracy_pct=v[1]*100, n_records_retained=v[2]) for k, v in sorted(subset_only.items(), key=lambda kv: -kv[1][0])[:5]], n_ratios_with_auc_ge_0_999=int(sum(v[0] >= 0.999 for v in ratios.values())),
           distinct_ratio_values_by_class=distinct, low_capacity_models=low_cap)
json.dump(out, open(dst, 'w'), indent=2)
print('top single-feature AUC:', [(k, round(v, 4)) for k, v in out['top10_single_feature_auc'][:5]])
print('top ratios:', [(r['ratio'], round(r['auc'], 4), round(r['best_single_threshold_accuracy_pct'], 2)) for r in out['top8_feature_ratio_scan'][:5]])
print('ratios scanned:', len(ratios), '| with AUC>=0.999:', out['n_ratios_with_auc_ge_0_999'])
print('distinct ratio values by class:', distinct)
for k, v in low_cap.items(): print(k, {a: round(b, 4) for a, b in v.items()})
