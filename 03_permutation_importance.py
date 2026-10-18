"""
03_permutation_importance.py  (version 3)
Permutation importance for the tuned KNN model, recomputed on the corrected
dataset.  The five splits (seeds 77-81) are REPEATED PARTITIONS OF THE SAME
DATASET, not independent traffic datasets, so the spread across them reflects
partition sensitivity only.  K is selected by inner 5-fold CV on each
training partition; importance is measured on the held-out 20% (10 repeats).
Because several predictors are correlated or arithmetically related (packet
counts, header lengths, byte totals, rates), single-feature permutation
importance can be diluted across correlated partners.  This version therefore
also reports (i) feature pairs with |Spearman| >= 0.90 and (ii) GROUPED
permutation importance, where every member of a correlated group is permuted
jointly with the same row permutation.  Conclusions apply to the evaluated
KNN model and the synthetic distributions only.
"""
import json, os, numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
KNN_GRID = [1, 3, 5, 7, 9, 11]; SEEDS = [77, 78, 79, 80, 81]; REPEATS = 10

def select_k(X, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); sc = {k: [] for k in KNN_GRID}
    for a, b in skf.split(X, y):
        s = MinMaxScaler().fit(X[a]); Xa, Xb = s.transform(X[a]), s.transform(X[b])
        for k in KNN_GRID:
            sc[k].append((KNeighborsClassifier(k, algorithm='kd_tree').fit(Xa, y[a]).predict(Xb) == y[b]).mean())
    m = {k: np.mean(v) for k, v in sc.items()}; return max(m, key=m.get)

def grouped_importance(model, X, y, groups, rng, repeats):
    base = (model.predict(X) == y).mean(); out = {}
    for g, cols in groups.items():
        d = []
        for _ in range(repeats):
            Xp = X.copy(); p = rng.permutation(len(X)); Xp[:, cols] = X[p][:, cols]
            d.append(base - (model.predict(Xp) == y).mean())
        out[g] = float(np.mean(d))
    return out

def main():
    df = pd.read_csv(os.path.join(HERE, 'network_flood_dataset.csv'))
    feats = [c for c in df.columns if c != 'Label']; X = df[feats].values.astype(float); y = df.Label.values
    corr = df[feats].corr(method='spearman').abs().fillna(0).values
    pairs = [(feats[i], feats[j], float(corr[i, j])) for i in range(len(feats)) for j in range(i + 1, len(feats)) if corr[i, j] >= 0.9]
    pairs.sort(key=lambda t: -t[2])
    dist = 1 - corr; np.fill_diagonal(dist, 0)
    cl = fcluster(linkage(squareform(dist, checks=False), 'average'), t=0.10, criterion='distance')   # |rho| >= 0.90 on average
    groups = {}
    for f, c in zip(feats, cl): groups.setdefault(int(c), []).append(f)
    groups = {'+'.join(v): [feats.index(f) for f in v] for v in groups.values()}
    imp = {f: [] for f in feats}; gimp = {g: [] for g in groups}; ks = []; vecs = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        k = select_k(Xtr, ytr, seed); ks.append(k)
        s = MinMaxScaler().fit(Xtr); m = KNeighborsClassifier(k, algorithm='kd_tree').fit(s.transform(Xtr), ytr)
        Xt = s.transform(Xte)
        r = permutation_importance(m, Xt, yte, n_repeats=REPEATS, random_state=seed, scoring='accuracy')
        for i, f in enumerate(feats): imp[f].append(float(r.importances_mean[i]))
        vecs.append(r.importances_mean)
        gi = grouped_importance(m, Xt, yte, groups, np.random.default_rng(seed), REPEATS)
        for g, v in gi.items(): gimp[g].append(v)
    summ = {f: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1))) for f, v in imp.items()}
    ranked = sorted(summ.items(), key=lambda kv: -kv[1]['mean'])
    top5 = {f: 0 for f in feats}
    for i in range(len(SEEDS)):
        for f in sorted(feats, key=lambda f: -imp[f][i])[:5]: top5[f] += 1
    rho = [stats.spearmanr(vecs[i], vecs[j])[0] for i in range(5) for j in range(i + 1, 5)]
    gsumm = sorted(({'group': g, 'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)), 'size': len(groups[g])} for g, v in gimp.items()), key=lambda d: -d['mean'])
    out = dict(selected_k_per_split=ks, seeds=SEEDS, note='five repeated partitions of the same dataset (not independent datasets)',
               importance_mean_std=summ, ranked_top10=ranked[:10], top5_frequency=top5,
               mean_pairwise_spearman_between_splits=float(np.mean(rho)), min_pairwise_spearman=float(np.min(rho)),
               correlated_pairs_abs_spearman_ge_0_90=pairs, correlated_groups=[list(g.split('+')) for g in groups if '+' in g],
               grouped_importance_ranked=gsumm[:10])
    json.dump(out, open(os.path.join(HERE, 'permutation_importance_results.json'), 'w'), indent=2)
    print('K per split:', ks); print('mean pairwise Spearman between splits: %.3f (min %.3f)' % (np.mean(rho), np.min(rho)))
    for n, s in ranked[:10]: print(f'  {n:18s} {s["mean"]:.4f} ± {s["std"]:.4f}  top5={top5[n]}/5')
    print('correlated pairs (|rho|>=0.9):', [(a, b, round(c, 3)) for a, b, c in pairs])
    print('grouped:', [(d['group'], round(d['mean'], 4)) for d in gsumm[:6]])

if __name__ == '__main__':
    main()
