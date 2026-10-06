"""
03_permutation_importance.py

Round-2 addition: assesses the STABILITY of the permutation-importance
ranking (Reviewer 1, Round 1: "the stability of the permutation-importance
results should be discussed"), and keeps the K-selection procedure
consistent with the nested-CV design used for the main comparison
(Reviewer 2, Round 2).

For each of 5 independent random splits (seeds 77, 78, 79, 80, 81):
  1. Stratified 80:20 train-test split.
  2. K is selected via 5-fold inner CV on the 80% training partition only
     (same procedure as the outer/inner design in
     02_nested_cv_evaluation.py) -- the 20% held-out partition is never
     used for K selection.
  3. KNN(K=selected) is fit on the scaled training partition.
  4. Permutation importance (Altmann et al., 2010) is computed on the
     held-out 20% test partition, n_repeats=10.

The mean and standard deviation of each feature's importance across the 5
independent repeats is reported, which is what lets us describe the
ranking's stability rather than treating a single run as definitive.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(__file__)
KNN_GRID = [1, 3, 5, 7, 9, 11]
SEEDS = [77, 78, 79, 80, 81]


def select_k(X_tr, y_tr, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {k: [] for k in KNN_GRID}
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        Xi_tr, Xi_val = X_tr[tr_idx], X_tr[val_idx]
        yi_tr, yi_val = y_tr[tr_idx], y_tr[val_idx]
        scaler = MinMaxScaler().fit(Xi_tr)
        Xi_tr_n, Xi_val_n = scaler.transform(Xi_tr), scaler.transform(Xi_val)
        for k in KNN_GRID:
            m = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", metric="euclidean")
            m.fit(Xi_tr_n, yi_tr)
            pred = m.predict(Xi_val_n)
            scores[k].append((pred == yi_val).mean())
    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(mean_scores, key=mean_scores.get)


def main():
    df = pd.read_csv(os.path.join(HERE, "network_flood_dataset.csv"))
    feature_names = [c for c in df.columns if c != "Label"]
    X = df[feature_names].values
    y = df["Label"].values

    all_importances = {f: [] for f in feature_names}
    selected_ks = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        k = select_k(X_tr, y_tr, seed)
        selected_ks.append(k)

        scaler = MinMaxScaler().fit(X_tr)
        X_tr_n, X_te_n = scaler.transform(X_tr), scaler.transform(X_te)

        model = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", metric="euclidean")
        model.fit(X_tr_n, y_tr)

        result = permutation_importance(
            model, X_te_n, y_te, n_repeats=10, random_state=seed, scoring="accuracy"
        )
        for i, f in enumerate(feature_names):
            all_importances[f].append(float(result.importances_mean[i]))

    summary = {}
    for f in feature_names:
        vals = np.array(all_importances[f])
        summary[f] = {"mean": float(vals.mean()), "std": float(vals.std())}

    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean"], reverse=True)

    # Rank stability: how often was each feature in the top 5 across the 5 repeats?
    top5_counts = {f: 0 for f in feature_names}
    for i, seed in enumerate(SEEDS):
        run_scores = {f: all_importances[f][i] for f in feature_names}
        top5 = sorted(run_scores, key=run_scores.get, reverse=True)[:5]
        for f in top5:
            top5_counts[f] += 1

    output = {
        "selected_k_per_repeat": selected_ks,
        "seeds": SEEDS,
        "importance_mean_std": summary,
        "ranked_top10": ranked[:10],
        "top5_frequency_across_5_repeats": top5_counts,
    }
    out_path = os.path.join(HERE, "permutation_importance_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    print("Selected K per repeat:", selected_ks)
    print("\nTop 10 features (mean permutation importance across 5 repeats):")
    for name, s in ranked[:10]:
        print(f"  {name:20s} mean={s['mean']:.4f}  std={s['std']:.4f}  top5_freq={top5_counts[name]}/5")


if __name__ == "__main__":
    main()
