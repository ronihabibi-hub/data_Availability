"""
02_nested_cv_evaluation.py

Round-2 revision pipeline. Implements a NESTED stratified cross-validation
design so that hyperparameter selection for ALL THREE models (KNN,
Decision Tree, Random Forest) happens only on each outer-training fold,
while the outer-test fold is never touched during selection -- directly
addressing Reviewer 2's Round-2 comment that "K tuning should occur only
in the inner CV of each outer-training fold, while the outer-test fold
remains completely unseen," and Reviewer 1's Round-2 comment that the
KNN-vs-tree comparison was not "completely balanced" because only KNN was
tuned.

Design
------
Outer loop : 5-fold stratified CV (same 5-fold split used for the
             comparative evaluation reported in the manuscript).
Inner loop : 5-fold stratified CV, run ONLY on the outer-training fold,
             used to pick each model's hyperparameter for that outer fold.

Hyperparameter grids (kept small and standard, not the point of the study):
  KNN            : n_neighbors in {1, 3, 5, 7, 9, 11}, Euclidean distance
  Decision Tree  : max_depth in {None, 5, 10, 15, 20}
  Random Forest  : n_estimators in {50, 100, 200}, other params default

For every outer fold and every model we record:
  - the selected hyperparameter
  - accuracy, precision, recall, F1 on the (unseen) outer-test fold
  - training time (wall clock, seconds)
  - inference latency, measured as: fit on outer-train -> 3 untimed
    warm-up prediction passes over the full outer-test fold -> 10 timed
    repeated full-batch prediction passes -> mean wall-clock time per
    pass, normalized to ms per 1,000 flow records (NOT "packets" -- the
    dataset is flow-level, so the unit is redefined here per Reviewer 1 /
    Reviewer 2's Round-2 comment).
  - peak memory allocated during model.fit (tracemalloc, KB)
  - serialized (pickled) model size (KB)

Aggregated out-of-fold predictions across the 5 outer folds are used for
ROC-AUC and the confusion matrix, exactly as in the Round-1 protocol.
95% confidence intervals for each metric are computed across the 5
outer-fold values using the t-distribution (n=5, df=4).
"""

import json
import os
import pickle
import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(__file__)
SEED = 77
N_OUTER = 5
N_INNER = 5
N_WARMUP = 3
N_TIMING_REPEATS = 10

KNN_GRID = [1, 3, 5, 7, 9, 11]
DT_GRID = [None, 5, 10, 15, 20]
RF_GRID = [50, 100, 200]


def make_model(name, param):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=param, algorithm="kd_tree", metric="euclidean")
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=param, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=param, random_state=SEED, n_jobs=1)
    raise ValueError(name)


def inner_select(name, grid, X_tr, y_tr):
    """Inner 5-fold stratified CV on the outer-training fold only."""
    inner_skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    scores = {p: [] for p in grid}
    for in_tr_idx, in_val_idx in inner_skf.split(X_tr, y_tr):
        Xi_tr, Xi_val = X_tr[in_tr_idx], X_tr[in_val_idx]
        yi_tr, yi_val = y_tr[in_tr_idx], y_tr[in_val_idx]
        scaler = MinMaxScaler().fit(Xi_tr)
        Xi_tr_n = scaler.transform(Xi_tr)
        Xi_val_n = scaler.transform(Xi_val)
        for p in grid:
            m = make_model(name, p)
            m.fit(Xi_tr_n, yi_tr)
            pred = m.predict(Xi_val_n)
            scores[p].append(accuracy_score(yi_val, pred))
    mean_scores = {p: float(np.mean(v)) for p, v in scores.items()}
    best_param = max(mean_scores, key=mean_scores.get)
    return best_param, mean_scores


def time_inference(model, X_test):
    for _ in range(N_WARMUP):
        model.predict(X_test)
    durations = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        model.predict(X_test)
        durations.append(time.perf_counter() - t0)
    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))
    n = len(X_test)
    ms_per_1000 = (mean_duration / n) * 1000 * 1000  # seconds/record -> ms/record -> ms/1000 records
    ms_per_1000_std = (std_duration / n) * 1000 * 1000
    throughput_rps = n / mean_duration
    return ms_per_1000, ms_per_1000_std, throughput_rps


def ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    sem = stats.sem(values)
    h = sem * stats.t.ppf(0.975, n - 1)
    return mean, mean - h, mean + h


def main():
    df = pd.read_csv(os.path.join(HERE, "network_flood_dataset.csv"))
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    models = {"KNN": KNN_GRID, "DecisionTree": DT_GRID, "RandomForest": RF_GRID}

    fold_records = {name: [] for name in models}
    agg_true = {name: [] for name in models}
    agg_pred = {name: [] for name in models}
    selected_params_per_fold = {name: [] for name in models}

    for fold_i, (train_idx, test_idx) in enumerate(outer_skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, grid in models.items():
            # --- inner CV: select hyperparameter using outer-TRAIN only ---
            best_param, inner_scores = inner_select(name, grid, X_train, y_train)
            selected_params_per_fold[name].append(best_param)

            # --- refit on the full outer-training fold with the selected param ---
            scaler = MinMaxScaler().fit(X_train)
            X_train_n = scaler.transform(X_train)
            X_test_n = scaler.transform(X_test)

            model = make_model(name, best_param)

            tracemalloc.start()
            t0 = time.perf_counter()
            model.fit(X_train_n, y_train)
            train_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            y_pred = model.predict(X_test_n)
            ms_per_1000, ms_per_1000_std, throughput = time_inference(model, X_test_n)
            model_size_kb = len(pickle.dumps(model)) / 1024.0

            fold_records[name].append({
                "fold": fold_i,
                "selected_param": best_param,
                "inner_cv_scores": inner_scores,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "train_time_s": train_time,
                "latency_ms_per_1000_records": ms_per_1000,
                "latency_ms_per_1000_records_std": ms_per_1000_std,
                "throughput_records_per_s": throughput,
                "peak_fit_memory_kb": peak_mem / 1024.0,
                "model_size_kb": model_size_kb,
            })
            agg_true[name].extend(y_test.tolist())
            agg_pred[name].extend(y_pred.tolist())

    # ---------------- Aggregate + summarize ----------------
    summary = {}
    for name in models:
        recs = fold_records[name]
        def col(k):
            return [r[k] for r in recs]

        acc_m, acc_lo, acc_hi = ci95(col("accuracy"))
        prec_m, prec_lo, prec_hi = ci95(col("precision"))
        rec_m, rec_lo, rec_hi = ci95(col("recall"))
        f1_m, f1_lo, f1_hi = ci95(col("f1"))
        lat_m, lat_lo, lat_hi = ci95(col("latency_ms_per_1000_records"))
        tt_m, tt_lo, tt_hi = ci95(col("train_time_s"))
        thr_m, thr_lo, thr_hi = ci95(col("throughput_records_per_s"))
        mem_m, mem_lo, mem_hi = ci95(col("peak_fit_memory_kb"))
        size_m, size_lo, size_hi = ci95(col("model_size_kb"))

        y_true = np.array(agg_true[name])
        y_pred = np.array(agg_pred[name])
        overall_auc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()

        summary[name] = {
            "selected_param_per_fold": selected_params_per_fold[name],
            "accuracy_pct": {"mean": acc_m * 100, "std": float(np.std(col("accuracy"))) * 100,
                              "ci95_lo": acc_lo * 100, "ci95_hi": acc_hi * 100},
            "precision_pct": {"mean": prec_m * 100, "std": float(np.std(col("precision"))) * 100,
                                "ci95_lo": prec_lo * 100, "ci95_hi": prec_hi * 100},
            "recall_pct": {"mean": rec_m * 100, "std": float(np.std(col("recall"))) * 100,
                             "ci95_lo": rec_lo * 100, "ci95_hi": rec_hi * 100},
            "f1_pct": {"mean": f1_m * 100, "std": float(np.std(col("f1"))) * 100,
                        "ci95_lo": f1_lo * 100, "ci95_hi": f1_hi * 100},
            "roc_auc": overall_auc,
            "confusion_matrix_TN_FP_FN_TP": [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
            "latency_ms_per_1000_records": {"mean": lat_m, "std": float(np.std(col("latency_ms_per_1000_records"))),
                                              "ci95_lo": lat_lo, "ci95_hi": lat_hi},
            "train_time_s": {"mean": tt_m, "std": float(np.std(col("train_time_s"))),
                               "ci95_lo": tt_lo, "ci95_hi": tt_hi},
            "throughput_records_per_s": {"mean": thr_m, "ci95_lo": thr_lo, "ci95_hi": thr_hi},
            "peak_fit_memory_kb": {"mean": mem_m, "ci95_lo": mem_lo, "ci95_hi": mem_hi},
            "model_size_kb": {"mean": size_m, "ci95_lo": size_lo, "ci95_hi": size_hi},
            "fold_level": recs,
        }

    # ---------------- Statistical comparisons across models (per-fold paired) ----------------
    def paired_vals(name, key):
        return np.array([r[key] for r in fold_records[name]])

    stat_tests = {}
    pairs = [("KNN", "DecisionTree"), ("KNN", "RandomForest"), ("DecisionTree", "RandomForest")]
    for metric in ["accuracy", "latency_ms_per_1000_records"]:
        stat_tests[metric] = {}
        for a, b in pairs:
            va = paired_vals(a, metric)
            vb = paired_vals(b, metric)
            try:
                w_stat, w_p = stats.wilcoxon(va, vb)
            except Exception as e:
                w_stat, w_p = None, None
            stat_tests[metric][f"{a}_vs_{b}"] = {"wilcoxon_stat": w_stat, "p_value": w_p}
        # Friedman test across all three
        va = paired_vals("KNN", metric)
        vb = paired_vals("DecisionTree", metric)
        vc = paired_vals("RandomForest", metric)
        f_stat, f_p = stats.friedmanchisquare(va, vb, vc)
        stat_tests[metric]["friedman_all_three"] = {"statistic": f_stat, "p_value": f_p}

    output = {
        "config": {
            "outer_folds": N_OUTER,
            "inner_folds": N_INNER,
            "seed": SEED,
            "warmup_passes": N_WARMUP,
            "timed_repeats": N_TIMING_REPEATS,
            "knn_grid": KNN_GRID,
            "decision_tree_max_depth_grid": DT_GRID,
            "random_forest_n_estimators_grid": RF_GRID,
        },
        "results": summary,
        "statistical_tests": stat_tests,
    }

    out_path = os.path.join(HERE, "nested_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: str(o))
    print(f"Wrote {out_path}")

    for name in models:
        s = summary[name]
        print(f"\n=== {name} ===")
        print("selected params per outer fold:", s["selected_param_per_fold"])
        print(f"Accuracy: {s['accuracy_pct']['mean']:.2f}% (95% CI {s['accuracy_pct']['ci95_lo']:.2f}-{s['accuracy_pct']['ci95_hi']:.2f})")
        print(f"Latency:  {s['latency_ms_per_1000_records']['mean']:.4f} ms/1000 records")
        print(f"Train time: {s['train_time_s']['mean']*1000:.3f} ms | Throughput: {s['throughput_records_per_s']['mean']:.0f} rec/s")
        print(f"Peak fit memory: {s['peak_fit_memory_kb']['mean']:.1f} KB | Model size: {s['model_size_kb']['mean']:.1f} KB")

    print("\n=== Statistical tests (accuracy, per-fold paired) ===")
    for pair, res in stat_tests["accuracy"].items():
        print(pair, res)
    print("\n=== Statistical tests (latency, per-fold paired) ===")
    for pair, res in stat_tests["latency_ms_per_1000_records"].items():
        print(pair, res)


if __name__ == "__main__":
    main()
