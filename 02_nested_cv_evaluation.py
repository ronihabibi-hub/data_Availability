"""
02_nested_cv_evaluation.py  (version 3, Round-3 revision)

Nested stratified cross-validation for KNN, Decision Tree and Random Forest.
All three models follow ONE shared nested-validation protocol; each has its
own, model-specific, search space (it is NOT claimed that the models are
"equally tuned").

Outer loop : 5-fold stratified CV (reporting).
Inner loop : 5-fold stratified CV on the outer-training fold only (selection,
             criterion = accuracy).  The outer-test fold is never used for
             selection.  MinMax scaling is fitted on training data only.

Changes in version 3 (editor, Round 3)
  * ROC-AUC (and PR-AUC) are computed from OUT-OF-FOLD CONTINUOUS SCORES
    (predict_proba of the flood class), not from hard class predictions.
  * Uncertainty: proportion metrics (accuracy, precision, recall) use Wilson
    score intervals on the pooled out-of-fold counts; F1, ROC-AUC and PR-AUC
    use percentile bootstrap over pooled out-of-fold records.  All intervals
    stay inside [0, 1] without truncation.  They describe finite-test-sample
    uncertainty of the out-of-fold predictions; they do not capture
    training-set variability and the folds share training data, so they are
    reported as descriptive uncertainty, not as formal inference.
    Fold-level mean, SD and range are reported alongside.
  * Model comparison: paired bootstrap of differences (same resampled
    records for every model), exact McNemar test on pooled out-of-fold
    predictions with Holm adjustment, odds ratio of discordant pairs, and an
    EXACT permutation version of the Friedman test over the 5 outer folds
    (no chi-square approximation).  Pairwise superiority is not inferred
    from the omnibus test.
  * Cost metrics: training time is measured WITHOUT tracemalloc; peak
    tracemalloc allocation is measured in a separate fit and the tracing
    overhead is reported.  tracemalloc starts after the input arrays exist,
    so it reports peak NEW Python/NumPy allocations during fit only.
    Serialized (pickle) size is reported separately.  Intervals for cost
    metrics are t-intervals on the log scale over the 5 fold values
    (cannot go below zero).  Latency is prediction on prepared feature arrays
    only (no capture, flow construction, feature extraction, scaling, alerts).
  * KNN backend is explicitly kd_tree (verified through model._fit_method).
"""
import itertools, json, os, pickle, platform, sys, time, tracemalloc
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 77
N_OUTER = N_INNER = 5
N_WARMUP, N_TIMING_REPEATS = 3, 10
N_BOOT = 2000
GRIDS = {"KNN": [1, 3, 5, 7, 9, 11], "DecisionTree": [None, 5, 10, 15, 20], "RandomForest": [50, 100, 200]}
PAIRS = [("KNN", "DecisionTree"), ("KNN", "RandomForest"), ("DecisionTree", "RandomForest")]


def make_model(name, p):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=p, algorithm="kd_tree", metric="euclidean")
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=p, random_state=SEED)
    return RandomForestClassifier(n_estimators=p, random_state=SEED, n_jobs=1)


def inner_select(name, grid, X_tr, y_tr):
    skf = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED)
    sc = {p: [] for p in grid}
    for a, b in skf.split(X_tr, y_tr):
        s = MinMaxScaler().fit(X_tr[a]); Xa, Xb = s.transform(X_tr[a]), s.transform(X_tr[b])
        for p in grid:
            sc[p].append(accuracy_score(y_tr[b], make_model(name, p).fit(Xa, y_tr[a]).predict(Xb)))
    ms = {p: float(np.mean(v)) for p, v in sc.items()}
    return max(ms, key=ms.get), ms


def time_inference(model, X):
    for _ in range(N_WARMUP): model.predict(X)
    d = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter(); model.predict(X); d.append(time.perf_counter() - t0)
    m = float(np.mean(d)); n = len(X)
    return m / n * 1e6, float(np.std(d)) / n * 1e6, n / m          # ms per 1,000 records, sd, records/s


def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def log_t_ci(v):
    v = np.asarray(v, float); l = np.log(v); m = l.mean(); h = stats.sem(l) * stats.t.ppf(.975, len(v) - 1)
    return float(np.exp(m)), float(np.exp(m - h)), float(np.exp(m + h))


def friedman_exact(M):
    """M: folds x models matrix. Exact permutation p-value of the Friedman statistic (within-fold permutations)."""
    n, k = M.shape
    def stat(A):
        R = np.apply_along_axis(stats.rankdata, 1, A)
        return 12.0 / (n * k * (k + 1)) * np.sum(R.sum(0) ** 2) - 3.0 * n * (k + 1)
    obs = stat(M); perms = list(itertools.permutations(range(k)))
    cnt = 0; tot = 0
    for combo in itertools.product(range(len(perms)), repeat=n):
        A = np.array([M[i, list(perms[c])] for i, c in enumerate(combo)])
        cnt += stat(A) >= obs - 1e-9; tot += 1
    return float(obs), cnt / tot


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def run_nested(df, timing=True, verbose=False):
    X = df.drop(columns=["Label"]).values.astype(float); y = df["Label"].values
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    folds = {m: [] for m in GRIDS}; oof_y = []; oof_pred = {m: [] for m in GRIDS}; oof_score = {m: [] for m in GRIDS}
    fold_idx = []
    for fi, (tr, te) in enumerate(outer.split(X, y), 1):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        oof_y.extend(yte.tolist()); fold_idx.extend([fi] * len(te))
        for name, grid in GRIDS.items():
            best, inner = inner_select(name, grid, Xtr, ytr)
            s = MinMaxScaler().fit(Xtr); Xtr_n, Xte_n = s.transform(Xtr), s.transform(Xte)
            rec = {"fold": fi, "selected_param": best, "inner_cv_scores": inner}
            if timing:
                t0 = time.perf_counter(); model = make_model(name, best).fit(Xtr_n, ytr); rec["train_time_s"] = time.perf_counter() - t0
                tracemalloc.start(); t0 = time.perf_counter(); m2 = make_model(name, best).fit(Xtr_n, ytr)
                rec["train_time_traced_s"] = time.perf_counter() - t0
                rec["peak_traced_alloc_kb"] = tracemalloc.get_traced_memory()[1] / 1024.0; tracemalloc.stop()
                rec["model_size_kb"] = len(pickle.dumps(model)) / 1024.0
                if name == "KNN": rec["knn_backend"] = model._fit_method
                lat, lat_sd, thr = time_inference(model, Xte_n)
                rec.update(latency_ms_per_1000_records=lat, latency_ms_per_1000_records_sd=lat_sd, throughput_records_per_s=thr)
            else:
                model = make_model(name, best).fit(Xtr_n, ytr)
            pred = model.predict(Xte_n); sc = model.predict_proba(Xte_n)[:, 1]
            rec.update(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                       recall=recall_score(yte, pred, zero_division=0), f1=f1_score(yte, pred, zero_division=0),
                       roc_auc=roc_auc_score(yte, sc), pr_auc=average_precision_score(yte, sc))
            folds[name].append(rec); oof_pred[name].extend(pred.tolist()); oof_score[name].extend(sc.tolist())
            if verbose: print(f"fold {fi} {name} param={best} acc={rec['accuracy']:.4f} auc={rec['roc_auc']:.4f}", flush=True)
    return dict(folds=folds, y=np.array(oof_y), pred={m: np.array(v) for m, v in oof_pred.items()},
                score={m: np.array(v) for m, v in oof_score.items()}, fold_idx=np.array(fold_idx))


def summarize(res, timing=True, rng_seed=SEED):
    y = res["y"]; n = len(y); out = {}
    rng = np.random.default_rng(rng_seed)
    boot = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    def metrics_on(idx, m):
        yt, yp, ys = y[idx], res["pred"][m][idx], res["score"][m][idx]
        if yt.min() == yt.max(): return None
        return dict(accuracy=accuracy_score(yt, yp), f1=f1_score(yt, yp, zero_division=0),
                    roc_auc=roc_auc_score(yt, ys), pr_auc=average_precision_score(yt, ys))
    bm = {m: [metrics_on(ix, m) for ix in boot] for m in GRIDS}
    for m in GRIDS:
        yp, ys = res["pred"][m], res["score"][m]
        tp = int(((y == 1) & (yp == 1)).sum()); fp = int(((y == 0) & (yp == 1)).sum())
        fn = int(((y == 1) & (yp == 0)).sum()); tn = int(((y == 0) & (yp == 0)).sum())
        fr = res["folds"][m]
        def fc(k): return np.array([r[k] for r in fr])
        def pct(a, lo, hi): return dict(estimate=a * 100, ci95_lo=lo * 100, ci95_hi=hi * 100)
        def bci(key):
            v = np.array([b[key] for b in bm[m]]); return np.percentile(v, 2.5), np.percentile(v, 97.5)
        acc_lo, acc_hi = wilson(tp + tn, n); pre_lo, pre_hi = wilson(tp, tp + fp); rec_lo, rec_hi = wilson(tp, tp + fn)
        pf = 2 * tp / (2 * tp + fp + fn)
        o = dict(
            selected_param_per_fold=[r["selected_param"] for r in fr],
            confusion_matrix_TN_FP_FN_TP=[tn, fp, fn, tp],
            pooled_oof=dict(
                accuracy=pct((tp + tn) / n, acc_lo, acc_hi), precision=pct(tp / (tp + fp), pre_lo, pre_hi),
                recall=pct(tp / (tp + fn), rec_lo, rec_hi), f1=pct(pf, *bci("f1")),
                roc_auc=dict(estimate=roc_auc_score(y, ys), ci95_lo=bci("roc_auc")[0], ci95_hi=bci("roc_auc")[1]),
                pr_auc=dict(estimate=average_precision_score(y, ys), ci95_lo=bci("pr_auc")[0], ci95_hi=bci("pr_auc")[1])),
            fold_level={k: dict(mean=float(fc(k).mean()), sd=float(fc(k).std(ddof=1)), min=float(fc(k).min()), max=float(fc(k).max()))
                        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]},
            fold_values={k: fc(k).tolist() for k in ["accuracy", "roc_auc"]},
            fold_records=fr)
        if timing:
            o["cost"] = {}
            for k in ["train_time_s", "train_time_traced_s", "latency_ms_per_1000_records", "throughput_records_per_s", "peak_traced_alloc_kb", "model_size_kb"]:
                mean, lo, hi = log_t_ci(fc(k)); o["cost"][k] = dict(geo_mean=mean, ci95_lo=lo, ci95_hi=hi, arith_mean=float(fc(k).mean()), min=float(fc(k).min()), max=float(fc(k).max()))
            o["cost"]["tracing_overhead_factor"] = float(np.mean(fc("train_time_traced_s") / fc("train_time_s")))
            if m == "KNN": o["knn_backend"] = sorted({r["knn_backend"] for r in fr})
        out[m] = o
    # ---- model comparison ----
    comp = {}; mc_p = []
    for a, b in PAIRS:
        ca = res["pred"][a] == y; cb = res["pred"][b] == y
        b01 = int((ca & ~cb).sum()); b10 = int((~ca & cb).sum())          # a right/b wrong ; a wrong/b right
        pm = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 > 0 else 1.0
        d = {k: np.array([bb[k] for bb in bm[a]]) - np.array([bb[k] for bb in bm[b]]) for k in ["accuracy", "f1", "roc_auc"]}
        pt = {k: float(({"accuracy": accuracy_score(y, res["pred"][a]) - accuracy_score(y, res["pred"][b]),
                         "f1": f1_score(y, res["pred"][a]) - f1_score(y, res["pred"][b]),
                         "roc_auc": roc_auc_score(y, res["score"][a]) - roc_auc_score(y, res["score"][b])})[k]) for k in d}
        comp[f"{a}_vs_{b}"] = dict(
            discordant_a_right_b_wrong=b01, discordant_a_wrong_b_right=b10,
            odds_ratio_discordant=(b01 / b10 if b10 else None), mcnemar_exact_p=pm,
            diff=dict({k: dict(estimate=pt[k], ci95_lo=float(np.percentile(d[k], 2.5)), ci95_hi=float(np.percentile(d[k], 97.5))) for k in d}))
        mc_p.append(pm)
    for (key, adj) in zip(comp, holm(mc_p)): comp[key]["mcnemar_holm_p"] = adj
    omni = {}
    for k in ["accuracy", "roc_auc"] + (["latency_ms_per_1000_records", "train_time_s"] if timing else []):
        M = np.column_stack([[r[k] for r in res["folds"][m]] for m in GRIDS])
        stat, p = friedman_exact(M); omni[k] = dict(friedman_statistic=stat, exact_permutation_p=p)
    if timing:
        ratios = {}
        for a, b in PAIRS:
            r = np.array([x["latency_ms_per_1000_records"] for x in res["folds"][a]]) / np.array([x["latency_ms_per_1000_records"] for x in res["folds"][b]])
            ratios[f"{a}/{b}"] = dict(zip(["geo_mean", "ci95_lo", "ci95_hi"], log_t_ci(r)), per_fold=r.tolist())
        comp["latency_ratio_per_fold"] = ratios
    return out, comp, omni


def environment():
    cpu = ""
    try:
        cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception: pass
    return dict(python=sys.version.split()[0], sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__,
                pandas=pd.__version__, platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count())


def main():
    df = pd.read_csv(os.path.join(HERE, "network_flood_dataset.csv"))
    res = run_nested(df, timing=True, verbose=True)
    summary, comp, omni = summarize(res, timing=True)
    out = dict(config=dict(outer_folds=N_OUTER, inner_folds=N_INNER, seed=SEED, warmup=N_WARMUP, timed_passes=N_TIMING_REPEATS,
                           batch_size_records=int(len(df) / N_OUTER), grids={k: [str(x) for x in v] for k, v in GRIDS.items()},
                           bootstrap_replicates=N_BOOT, environment=environment()),
               results=summary, pairwise=comp, omnibus_fold_level=omni)
    json.dump(out, open(os.path.join(HERE, "nested_cv_results.json"), "w"), indent=2, default=str)
    oof = pd.DataFrame({"record_index_in_outer_order": np.arange(len(res["y"])), "outer_fold": res["fold_idx"], "label": res["y"]})
    for m in GRIDS:
        oof[m + "_pred"] = res["pred"][m]; oof[m + "_flood_probability"] = res["score"][m]
    oof.to_csv(os.path.join(HERE, "oof_predictions.csv"), index=False)
    # K-tuning table (mean inner-CV accuracy per K across outer folds)
    kt = {str(k): float(np.mean([r["inner_cv_scores"][k] for r in res["folds"]["KNN"]])) for k in GRIDS["KNN"]}
    json.dump(kt, open(os.path.join(HERE, "k_tuning_table.json"), "w"), indent=2)
    for m, s in summary.items():
        p = s["pooled_oof"]
        print(f"\n{m}: params={s['selected_param_per_fold']} acc={p['accuracy']['estimate']:.2f} [{p['accuracy']['ci95_lo']:.2f},{p['accuracy']['ci95_hi']:.2f}] "
              f"AUC={p['roc_auc']['estimate']:.4f} [{p['roc_auc']['ci95_lo']:.4f},{p['roc_auc']['ci95_hi']:.4f}] F1={p['f1']['estimate']:.2f}")
        c = s["cost"]; print(f"  latency={c['latency_ms_per_1000_records']['geo_mean']:.3f} ms/1000  train={c['train_time_s']['geo_mean']*1000:.1f} ms  "
                             f"peakmem={c['peak_traced_alloc_kb']['geo_mean']:.0f} KB  size={c['model_size_kb']['geo_mean']:.0f} KB  overhead x{c['tracing_overhead_factor']:.2f}")
    print(json.dumps(comp, indent=1, default=str)[:2500]); print(omni)


if __name__ == "__main__":
    main()
