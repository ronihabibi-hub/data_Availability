"""
04_separability_analysis.py

Round-2 addition, directly answering Reviewer 1's Round-1 comment ("the
authors should investigate whether the generated normal and flood classes
are intrinsically highly separable") and Reviewer 2's related comment
about disclosing threats to validity from synthetic-data separability.

Two simple, transparent diagnostics:
 (a) Single-feature ROC-AUC for each of the top permutation-importance
     features, computed by thresholding that one feature alone -- shows
     how much separability already exists along individual axes, without
     any classifier.
 (b) A minimal-feature classifier (Decision Tree restricted to depth=1,
     i.e., a single-feature single-split "decision stump") fit on the
     full dataset -- shows how much accuracy a trivial one-rule detector
     already achieves, which is the most direct evidence of how
     separable the synthetic classes are.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(__file__)
SEED = 77

df = pd.read_csv(os.path.join(HERE, "network_flood_dataset.csv"))
y = df["Label"].values
feature_names = [c for c in df.columns if c != "Label"]

single_feature_auc = {}
for f in feature_names:
    x = df[f].values
    try:
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)  # direction-agnostic
    except Exception:
        auc = None
    single_feature_auc[f] = auc

ranked = sorted(single_feature_auc.items(), key=lambda kv: (kv[1] is not None, kv[1]), reverse=True)

# Decision stump (depth=1) 5-fold CV accuracy -- trivial one-rule detector
X = df[feature_names].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
stump_accs = []
for tr_idx, te_idx in skf.split(X, y):
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    stump.fit(X[tr_idx], y[tr_idx])
    stump_accs.append(stump.score(X[te_idx], y[te_idx]))

output = {
    "top10_single_feature_auc": ranked[:10],
    "decision_stump_depth1_5fold_cv_accuracy": {
        "mean_pct": float(np.mean(stump_accs)) * 100,
        "std_pct": float(np.std(stump_accs)) * 100,
        "fold_values_pct": [float(a) * 100 for a in stump_accs],
    },
}
with open(os.path.join(HERE, "separability_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Top single-feature AUCs:")
for name, auc in ranked[:6]:
    print(f"  {name:20s} AUC={auc:.4f}")
print(f"\nDecision stump (depth=1, single split) 5-fold CV accuracy: "
      f"{output['decision_stump_depth1_5fold_cv_accuracy']['mean_pct']:.2f}% "
      f"± {output['decision_stump_depth1_5fold_cv_accuracy']['std_pct']:.2f}%")
