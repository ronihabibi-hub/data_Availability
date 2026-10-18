"""
06_sensitivity_analysis.py  (new in version 3; editor point 8)
Re-runs the SAME nested-CV protocol (02_nested_cv_evaluation.py, without the
timing part) on regenerated datasets that vary one factor at a time:
  stealth fraction of floods (class overlap): 0.0 / 0.25 (baseline) / 0.5
  dataset size n                             : 1,000 / 2,500 (baseline) / 5,000
  benign application mix                     : baseline / lecture / exam / lab
  generator seed (replication)               : 77 (baseline) / 78 / 79
Output: sensitivity_results.json
"""
import importlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
gen = importlib.import_module('01_generate_dataset'); ev = importlib.import_module('02_nested_cv_evaluation')

SCEN = [('baseline', dict()),
        ('stealth_0.00', dict(stealth_frac=0.0)), ('stealth_0.50', dict(stealth_frac=0.5)),
        ('n_1000', dict(n_samples=1000)), ('n_5000', dict(n_samples=5000)),
        ('mix_lecture', dict(mix='lecture')), ('mix_exam', dict(mix='exam')), ('mix_lab', dict(mix='lab')),
        ('seed_78', dict(seed=78)), ('seed_79', dict(seed=79))]
out = {}
for name, kw in SCEN:
    t0 = time.time(); df, _ = gen.generate(**kw)
    res = ev.run_nested(df, timing=False); summ, comp, omni = ev.summarize(res, timing=False)
    out[name] = dict(params=kw, n=int(len(df)), flood_share=float(df.Label.mean()),
                     models={m: {k: summ[m]['pooled_oof'][k] for k in ['accuracy', 'f1', 'roc_auc']} for m in summ},
                     selected_params={m: summ[m]['selected_param_per_fold'] for m in summ},
                     pairwise_accuracy_diff={k: dict(v['diff']['accuracy'], mcnemar_holm_p=v['mcnemar_holm_p']) for k, v in comp.items()},
                     omnibus_accuracy_exact_p=omni['accuracy']['exact_permutation_p'])
    print(name, {m: (round(v['accuracy']['estimate'], 2), round(v['roc_auc']['estimate'], 4)) for m, v in out[name]['models'].items()}, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(HERE, 'sensitivity_results.json'), 'w'), indent=2, default=float)
