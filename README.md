# Reproducibility Package — Round 2 Revision

Comparative Evaluation of KNN, Decision Tree, and Random Forest Using Flow-Based
Features for Flood Attack Detection in a Synthetic Academic-Network Traffic Scenario

This package accompanies the Round 2 revised manuscript submitted to the
Journal of Information Systems and Informatics, and supports the reproducibility
requirement raised by Reviewer 2 in Round 2 ("Provide a public repository or
supplementary material containing the dataset-generation script and evaluation
code to ensure reproducibility").

## Contents

| File | Purpose |
|---|---|
| `01_generate_dataset.py` | Regenerates `network_flood_dataset.csv` from documented, seeded distribution parameters (seed = 77). Produces 2,500 records: 2,000 normal + 500 flood, 25 flow-based features + binary label. |
| `02_nested_cv_evaluation.py` | Runs the nested cross-validation design described in Methods §2.4-2.5: outer 5-fold CV for reporting, inner 5-fold CV (per outer-training fold) for hyperparameter selection of KNN, Decision Tree, and Random Forest. Computes accuracy/precision/recall/F1/ROC-AUC, 95% CIs, confusion matrices, training time, inference latency (ms per 1,000 flow records), throughput, peak fit memory, serialized model size, and Wilcoxon/Friedman statistical tests. Writes `nested_cv_results.json`. |
| `03_permutation_importance.py` | Runs the feature-importance stability check described in Methods §2.6: 5 independent stratified 80:20 splits, K selected by inner CV within each split, permutation importance computed on the held-out 20%. Writes `permutation_importance_results.json`. |
| `04_separability_analysis.py` | Runs the class-separability diagnostics described in Methods §2.7: single-feature ROC-AUC per feature, and 5-fold CV accuracy of a depth-1 decision stump. Writes `separability_results.json`. |
| `network_flood_dataset.csv` | The exact generated dataset used for the numbers reported in the manuscript (2,500 x 26, including `Label`). |
| `nested_cv_results.json`, `permutation_importance_results.json`, `separability_results.json`, `k_tuning_table.json` | Raw outputs from the scripts above, as used to populate Tables 2-7 and Figures 3-7 of the manuscript. |

## How to reproduce

```bash
pip install numpy pandas scikit-learn scipy
python3 01_generate_dataset.py          # regenerates network_flood_dataset.csv (deterministic, seed=77)
python3 02_nested_cv_evaluation.py      # Tables 2-5, statistical tests
python3 03_permutation_importance.py    # Table 6
python3 04_separability_analysis.py     # Table 7
```

## Environment used for the Round 2 numbers reported in the manuscript

- Python 3.11.15
- scikit-learn 1.8.0, scipy 1.17.1, numpy 2.4.4, pandas 3.0.2
- Shared cloud virtual machine, 2 vCPUs, ~7.8 GB RAM (no GPU)

This differs from the Google Colab standard runtime used for the original
(Round 1) submission. As documented in the manuscript's Threats to Validity
(§3.9), the change in hardware and in the latency-timing methodology (warm-up
passes + repeated batch timing, introduced in this revision) means the
**absolute** latency/training-time/throughput figures in this revision are not
directly comparable to the absolute figures reported in the Round 1 submission.
The **relative ordering** between the three models (KNN slowest at inference,
Decision Tree fastest) is what the manuscript treats as evidence, and that
ordering is consistent between the two submissions.

## Independent verification performed before packaging

Before packaging, all four scripts were re-run independently, in sequence, in a
clean copy of this exact folder:

- `01_generate_dataset.py` was confirmed to regenerate a byte-identical CSV
  (verified by hash comparison), confirming the seed=77 generation is fully
  deterministic and self-contained.
- `02_nested_cv_evaluation.py`, `03_permutation_importance.py`, and
  `04_separability_analysis.py` were confirmed to reproduce the same
  per-fold selected hyperparameters, the same accuracy/latency figures (to
  reported precision), and the same statistical-test results as the numbers
  quoted in the manuscript.

No number in Tables 2-7 or Figures 3-7 of the manuscript was hand-edited after
being produced by these scripts.

## Ethical and integrity notes

- The dataset is **entirely synthetic**; it was generated to statistically
  emulate academic-network traffic and is not, and does not claim to be, a
  capture of real network activity. This is stated explicitly in the
  manuscript's Methods and Threats to Validity sections.
- No results derived from any exploratory or since-abandoned analysis path
  (including an earlier, unused PCA step and an earlier noise-injection
  experiment that was excluded from this study on integrity grounds) are
  included anywhere in this package or in the manuscript. Every number here
  traces to exactly the four scripts listed above.
