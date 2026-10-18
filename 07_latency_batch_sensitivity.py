"""
07_latency_batch_sensitivity.py  (new in version 3; editor point 5)
Shows how far the latency comparison depends on batch size and on the KNN
search backend.  One model per type is fitted on 80% of the dataset
(KNN K=5, Decision Tree unrestricted depth, Random Forest 100 trees) and
prediction time on prepared, scaled feature arrays is measured for several
batch sizes.  KNN is timed with the kd_tree backend (used in all other
experiments) and with brute-force search for comparison.
Output: latency_batch_results.json  (ms per 1,000 records, mean of timed passes)
"""
import json, os, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); SEED = 77
df = pd.read_csv(os.path.join(HERE, 'network_flood_dataset.csv'))
X = df.drop(columns='Label').values.astype(float); y = df.Label.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
s = MinMaxScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
models = {'KNN_kd_tree': KNeighborsClassifier(5, algorithm='kd_tree'), 'KNN_brute': KNeighborsClassifier(5, algorithm='brute'),
          'DecisionTree': DecisionTreeClassifier(random_state=SEED), 'RandomForest': RandomForestClassifier(100, random_state=SEED, n_jobs=1)}
for m in models.values(): m.fit(Xtr, ytr)
out = {}
for bs in [1, 10, 100, 500]:
    out[str(bs)] = {}
    nb = len(Xte) // bs; batches = [Xte[i * bs:(i + 1) * bs] for i in range(nb)]
    reps = 3 if bs == 1 else 5
    for name, m in models.items():
        for b in batches[:3]: m.predict(b)                                     # warm-up
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for b in batches: m.predict(b)
            times.append(time.perf_counter() - t0)
        n_rec = nb * bs
        out[str(bs)][name] = dict(ms_per_1000_records=float(np.mean(times) / n_rec * 1e6), n_records_per_pass=int(n_rec), passes=reps)
    print(bs, {k: round(v['ms_per_1000_records'], 2) for k, v in out[str(bs)].items()}, flush=True)
json.dump(dict(train_records=int(len(Xtr)), test_records=int(len(Xte)), results=out), open(os.path.join(HERE, 'latency_batch_results.json'), 'w'), indent=2)
