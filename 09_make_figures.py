import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve

R3 = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(R3, "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 220})
nested = json.load(open(f"{R3}/nested_cv_results.json")); perm = json.load(open(f"{R3}/permutation_importance_results.json"))
sens = json.load(open(f"{R3}/sensitivity_results.json")); lat = json.load(open(f"{R3}/latency_batch_results.json"))
oof = pd.read_csv(f"{R3}/oof_predictions.csv")
MODELS = ["KNN", "DecisionTree", "RandomForest"]; LAB = {"KNN": "KNN", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest"}
HATCH = {"KNN": "//", "DecisionTree": "xx", "RandomForest": ".."}; GRAY = {"KNN": "0.35", "DecisionTree": "0.6", "RandomForest": "0.85"}
LS = {"KNN": "-", "DecisionTree": "--", "RandomForest": ":"}

# ---- Figure 3: confusion matrices + ROC ----
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
for ax, m in zip(axes[:3], MODELS):
    tn, fp, fn, tp = nested["results"][m]["confusion_matrix_TN_FP_FN_TP"]; cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold", color="white" if cm[i, j] > cm.max() * .6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred.\nNormal", "Pred.\nFlood"], fontsize=9); ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nNormal", "Actual\nFlood"], fontsize=9)
    ax.set_title(LAB[m], fontsize=11)
ax = axes[3]
for m in MODELS:
    f, t, _ = roc_curve(oof.label, oof[m + "_flood_probability"]); a = nested["results"][m]["pooled_oof"]["roc_auc"]["estimate"]
    ax.plot(f, t, LS[m], color="black", lw=1.6, label=f"{LAB[m]} (AUC {a:.3f})")
ax.plot([0, 1], [0, 1], color="0.6", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("ROC (out-of-fold probabilities)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_confusion_roc.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)

# ---- Figure 4: latency (batch 500, 95% CI) + batch-size/backend curves ----
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax = axes[0]; g = [nested["results"][m]["cost"]["latency_ms_per_1000_records"] for m in MODELS]
v = [x["geo_mean"] for x in g]; lo = [x["geo_mean"] - x["ci95_lo"] for x in g]; hi = [x["ci95_hi"] - x["geo_mean"] for x in g]
ax.bar(range(3), v, yerr=[lo, hi], capsize=6, color=[GRAY[m] for m in MODELS], edgecolor="black", hatch=[HATCH[m] for m in MODELS]); ax.set_yscale("log")
ax.set_xticks(range(3)); ax.set_xticklabels([LAB[m] for m in MODELS], fontsize=9); ax.set_ylabel("Latency (ms per 1,000 flow records, log)")
for i, x in enumerate(g): ax.text(i, x["ci95_hi"] * 1.2, f"{x['geo_mean']:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("(a) Batch of 500 records, geometric mean\nand 95% interval over 5 folds", fontsize=10)
ax = axes[1]; bs = ["1", "10", "100", "500"]
series = [("KNN_kd_tree", "KNN (kd-tree)", "-", "o"), ("KNN_brute", "KNN (brute force)", "-.", "s"), ("DecisionTree", "Decision Tree", "--", "^"), ("RandomForest", "Random Forest (100 trees)", ":", "D")]
for k, l, ls, mk in series: ax.plot([int(b) for b in bs], [lat["results"][b][k]["ms_per_1000_records"] for b in bs], ls, marker=mk, color="black", lw=1.5, label=l)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([1, 10, 100, 500]); ax.set_xticklabels(["1", "10", "100", "500"]); ax.set_xlabel("Prediction batch size (records per call)"); ax.set_ylabel("ms per 1,000 records (log)"); ax.legend(fontsize=8)
ax.set_title("(b) Effect of batch size and KNN search backend", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_latency.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)

# ---- Figure 5: training time, traced peak allocation, serialized size ----
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, (k, yl, sc) in zip(axes, [("train_time_s", "Training time (ms)", 1000), ("peak_traced_alloc_kb", "Peak traced allocation\nduring fit (KB)", 1), ("model_size_kb", "Serialized model size (KB)", 1)]):
    g = [nested["results"][m]["cost"][k] for m in MODELS]; v = [x["geo_mean"] * sc for x in g]
    lo = [max(0, (x["geo_mean"] - x["ci95_lo"]) * sc) for x in g]; hi = [(x["ci95_hi"] - x["geo_mean"]) * sc for x in g]
    ax.bar(range(3), v, yerr=[lo, hi], capsize=5, color=[GRAY[m] for m in MODELS], edgecolor="black", hatch=[HATCH[m] for m in MODELS]); ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels([LAB[m].replace(" ", "\n") for m in MODELS], fontsize=9); ax.set_ylabel(yl, fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_computational_cost.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)

# ---- Figure 6: permutation importance ----
r = perm["ranked_top10"]; names = [x[0] for x in r]; means = [x[1]["mean"] for x in r]; stds = [x[1]["std"] for x in r]
fig, ax = plt.subplots(figsize=(7.0, 4.2)); yp = np.arange(len(names))[::-1]
ax.barh(yp, means, xerr=stds, color="0.55", edgecolor="black", hatch="//", capsize=4); ax.set_yticks(yp); ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Mean permutation importance (accuracy drop), ± SD over 5 partitions"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_importance.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)

# ---- Figure 7: sensitivity ----
order = ["baseline", "stealth_0.00", "stealth_0.50", "n_1000", "n_5000", "mix_lecture", "mix_exam", "mix_lab", "seed_78", "seed_79"]
lab = ["Baseline", "No stealth\nfloods", "50% stealth\nfloods", "n = 1,000", "n = 5,000", "Lecture\nmix", "Exam\nmix", "Lab\nmix", "Seed 78", "Seed 79"]
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
off = {"KNN": -.2, "DecisionTree": 0, "RandomForest": .2}; mk = {"KNN": "o", "DecisionTree": "s", "RandomForest": "^"}
for ax, key, yl in [(axes[0], "accuracy", "Accuracy (%)"), (axes[1], "roc_auc", "ROC-AUC")]:
    for m in MODELS:
        x = np.arange(len(order)) + off[m]; e = [sens[s]["models"][m][key] for s in order]
        y = [d["estimate"] for d in e]; lo = [d["estimate"] - d["ci95_lo"] for d in e]; hi = [d["ci95_hi"] - d["estimate"] for d in e]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk[m], color="black", mfc="white" if m == "DecisionTree" else ("black" if m == "KNN" else "0.6"), capsize=3, ms=6, label=LAB[m], lw=1)
    ax.set_ylabel(yl); ax.grid(True, axis="y", ls=":", alpha=.5)
axes[0].legend(ncol=3, fontsize=9, loc="lower left"); axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(lab, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sensitivity.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)

# ---- Figures 1-2: flowcharts ----
def box(ax, x, y, w, h, text, fc="#E8EEF7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.2, edgecolor="black", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
def arrow(ax, x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="black"))
fig, ax = plt.subplots(figsize=(9.5, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
rows = [["Literature Review\n& Problem\nIdentification", "Campus-style\nPacket-level Flow\nSimulation (seeded)", "Consistency Checks\n& Shortcut\nDiagnostics"],
        ["Min-Max Scaling\n(fit on training\ndata only)", "Nested CV:\nInner-Fold Tuning\n(K / max_depth /\nn_estimators)", "Nested CV:\nOuter-Fold Evaluation\n(out-of-fold scores)"],
        ["Accuracy, AUC,\nIntervals & Paired\nComparisons", "Cost Measurement,\nImportance &\nSensitivity Runs", "Findings &\nScope of\nConclusions"]]
w, h = 2.7, 1.1; xs = [.3, 3.65, 7.0]; ys = [3.6, 2.0, .4]
for row, y in zip(rows, ys):
    for x, t in zip(xs, row): box(ax, x, y, w, h, t)
arrow(ax, xs[0] + w, ys[0] + h / 2, xs[1], ys[0] + h / 2); arrow(ax, xs[1] + w, ys[0] + h / 2, xs[2], ys[0] + h / 2)
arrow(ax, xs[2] + w / 2, ys[0], xs[2] + w / 2, ys[1] + h); arrow(ax, xs[2], ys[1] + h / 2, xs[1] + w, ys[1] + h / 2); arrow(ax, xs[1], ys[1] + h / 2, xs[0] + w, ys[1] + h / 2)
arrow(ax, xs[0] + w / 2, ys[1], xs[0] + w / 2, ys[2] + h); arrow(ax, xs[0] + w, ys[2] + h / 2, xs[1], ys[2] + h / 2); arrow(ax, xs[1] + w, ys[2] + h / 2, xs[2], ys[2] + h / 2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_flowchart.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)
fig, ax = plt.subplots(figsize=(11.5, 3.2)); ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.2); ax.axis("off")
steps = ["Min-Max\nscaling\n(fit on\ntraining data)", "Outer 5-fold\nstratified split", "Inner 5-fold CV\n(outer-train only)\nper model:\nKNN K,\nDT depth,\nRF n_estim.", "Refit each model\non full outer-\ntrain with the\nselected value", "Evaluate once on\nunseen outer-test:\nclass labels,\nflood probability,\ncost metrics", "Pool 5 outer folds:\nWilson / bootstrap\nintervals, paired\ncomparisons"]
w, h = 1.9, 1.9; x = .15; y = .65
for i, s in enumerate(steps):
    box(ax, x, y, w, h, s, fc="#EAF0FB" if i in (1, 2) else "#F2F2F2")
    if i < len(steps) - 1: arrow(ax, x + w, y + h / 2, x + w + .25, y + h / 2)
    x += w + .25
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", bbox_inches="tight", pad_inches=.2); plt.close(fig)
print("ok")
