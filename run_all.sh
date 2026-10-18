#!/usr/bin/env bash
# Regenerates every number, table and figure of the manuscript (about 15 minutes on 2 vCPUs).
set -e
python3 01_generate_dataset.py            # network_flood_dataset.csv + _metadata.csv (seed 77)
python3 05_validate_dataset.py            # validation_results.json (19 consistency checks)
python3 04_separability_analysis.py       # separability_results.json (shortcut diagnostics)
python3 02_nested_cv_evaluation.py        # nested_cv_results.json, k_tuning_table.json, oof_predictions.csv
python3 03_permutation_importance.py      # permutation_importance_results.json
python3 06_sensitivity_analysis.py        # sensitivity_results.json (10 scenarios)
python3 07_latency_batch_sensitivity.py   # latency_batch_results.json
python3 08_error_analysis.py              # error_analysis.json
python3 09_make_figures.py                # figures/
