# Reproducibility package, version 3.0 (release v3.0)

Comparative Evaluation of KNN, Decision Tree, and Random Forest Using Flow-Based Features for Flood
Attack Detection in a Synthetic Academic-Network Traffic Scenario (Journal of Information Systems and
Informatics). Corresponding author: Roni Habibi (roni.habibi@ulbi.ac.id).

Version 3.0 replaces all earlier files. The dataset generator was rewritten, so the dataset and every
result file differ from earlier versions and must not be mixed with them.

## What the data are
Simulated flow records for a campus-style network scenario. Each flow is simulated at packet level
(packet counts per direction, payload lengths, inter-arrival gaps, header sizes, TCP flag events), and
all 25 flow features are computed from those quantities, so ordering and conservation relations hold by
construction. Benign traffic mixes eight application profiles; flood traffic mixes SYN, UDP and HTTP
variants, 25% at low rate. Profile weights and parameters are modelling assumptions informed
qualitatively by published campus-traffic studies; they are not fitted to a captured trace.
`network_flood_dataset_metadata.csv` holds the profile of each record (never a model input).

## Files
| File | Purpose |
|---|---|
| `01_generate_dataset.py` | Generator (seed 77; options for size, low-rate share, benign mix, seed) |
| `05_validate_dataset.py` | 19 consistency checks, `validation_results.json` |
| `04_separability_analysis.py` | Shortcut diagnostics: single-feature AUC, scan of all feature ratios, low-capacity models, label-permutation baseline, `separability_results.json` |
| `02_nested_cv_evaluation.py` | Nested stratified CV (5 x 5), shared protocol with model-specific search spaces; ROC-AUC from out-of-fold `predict_proba`; Wilson and bootstrap intervals; paired comparisons; cost metrics; `nested_cv_results.json`, `oof_predictions.csv`, `k_tuning_table.json` |
| `03_permutation_importance.py` | KNN permutation importance on 5 repeated partitions, correlated pairs, grouped importance |
| `06_sensitivity_analysis.py` | Ten scenarios (overlap, size, traffic mix, seed) |
| `07_latency_batch_sensitivity.py` | Latency by batch size (1, 10, 100, 500) and KNN backend (kd_tree vs brute) |
| `08_error_analysis.py` | Out-of-fold errors by traffic profile |
| `09_make_figures.py` | Figures used in the manuscript |
| `run_all.sh`, `requirements.txt` | Run everything; package versions |
| `*.json`, `oof_predictions.csv`, `network_flood_dataset*.csv` | Outputs behind Tables 3 to 10 and Figures 3 to 7 |

## Reproduce
```bash
pip install -r requirements.txt
./run_all.sh
```
Statistical results are deterministic given the seeds. Timing results (latency, training time) vary
from run to run and across machines.

## Environment of the reported results
Python 3.11.15; scikit-learn 1.8.0; SciPy 1.17.1; NumPy 2.4.4; pandas 3.0.2; Linux x86_64 virtual machine,
Intel Xeon 2.10 GHz, 2 logical CPUs, about 7.8 GB RAM, no GPU, single-threaded prediction. Also recorded in
`nested_cv_results.json` (config.environment).

## Scope
Offline, synthetic, flood attacks only. Runtime conclusions apply to the tested KNN backend (kd_tree,
with a brute-force comparison), 25 features, 2,000 training records, the stated batch sizes and this
hardware.
