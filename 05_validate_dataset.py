"""
05_validate_dataset.py
Physical / mathematical consistency checks for the synthetic flow dataset
(editor point 2).  Each check reports the number of violating records.
Usage: python 05_validate_dataset.py [dataset.csv]
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'network_flood_dataset.csv')
d = pd.read_csv(path)
n = d.Tot_Fwd_Pkts + d.Tot_Bwd_Pkts
sec = d.Flow_Duration / 1e6
tol = 1e-6
checks = {}
def chk(name, bad): checks[name] = int(np.sum(bad))

chk('fwd_len_min<=mean<=max', ~((d.Fwd_Pkt_Len_Min <= d.Fwd_Pkt_Len_Mean + tol) & (d.Fwd_Pkt_Len_Mean <= d.Fwd_Pkt_Len_Max + tol)))
chk('bwd_len_min<=max', d.Bwd_Pkt_Len_Min > d.Bwd_Pkt_Len_Max)
chk('IAT_min<=mean<=max', ~((d.Flow_IAT_Min <= d.Flow_IAT_Mean + tol) & (d.Flow_IAT_Mean <= d.Flow_IAT_Max + tol)))
chk('IAT_max<=duration', d.Flow_IAT_Max > d.Flow_Duration + tol)
chk('duration==IAT_mean*(pkts-1)', np.abs(d.Flow_IAT_Mean * (n - 1) - d.Flow_Duration) > 1e-6 * d.Flow_Duration + 1e-3)
chk('TotLen_Fwd==Fwd_mean*Fwd_pkts', np.abs(d.Fwd_Pkt_Len_Mean * d.Tot_Fwd_Pkts - d.TotLen_Fwd_Pkts) > 1e-6 * (d.TotLen_Fwd_Pkts + 1))
chk('TotLen_Fwd within [pkts*min, pkts*max]', (d.TotLen_Fwd_Pkts < d.Tot_Fwd_Pkts * d.Fwd_Pkt_Len_Min - tol) | (d.TotLen_Fwd_Pkts > d.Tot_Fwd_Pkts * d.Fwd_Pkt_Len_Max + tol))
chk('TotLen_Bwd within [pkts*min, pkts*max]', (d.TotLen_Bwd_Pkts < d.Tot_Bwd_Pkts * d.Bwd_Pkt_Len_Min - tol) | (d.TotLen_Bwd_Pkts > d.Tot_Bwd_Pkts * d.Bwd_Pkt_Len_Max + tol))
chk('Bwd bytes>0 only if Bwd pkts>0', (d.Tot_Bwd_Pkts == 0) & (d.TotLen_Bwd_Pkts > 0))
chk('Flow_Pkts_s*duration==pkts', np.abs(d.Flow_Pkts_s * sec - n) > 1e-6 * n)
chk('Flow_Byts_s*duration==bytes', np.abs(d.Flow_Byts_s * sec - (d.TotLen_Fwd_Pkts + d.TotLen_Bwd_Pkts)) > 1e-6 * (d.TotLen_Fwd_Pkts + d.TotLen_Bwd_Pkts + 1))
chk('Fwd_Pkts_s+Bwd_Pkts_s==Flow_Pkts_s', np.abs(d.Fwd_Pkts_s + d.Bwd_Pkts_s - d.Flow_Pkts_s) > 1e-6 * d.Flow_Pkts_s)
chk('Down_Up_Ratio==Bwd_pkts/Fwd_pkts', np.abs(d.Down_Up_Ratio - d.Tot_Bwd_Pkts / d.Tot_Fwd_Pkts) > 1e-9)
chk('payload length <=1472 bytes', (d.Fwd_Pkt_Len_Max > 1472) | (d.Bwd_Pkt_Len_Max > 1472))
chk('header len multiple of 8,20,32 per packet and <=32/pkt', (d.Fwd_Header_Len > 32 * d.Tot_Fwd_Pkts) | (d.Bwd_Header_Len > 32 * d.Tot_Bwd_Pkts) | (d.Fwd_Header_Len < 8 * d.Tot_Fwd_Pkts))
chk('flag counts <= packets', (d[['SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt']].max(axis=1) > n))
chk('URG in {0,1}', ~d.URG_Flag_Cnt.isin([0, 1]))
chk('no missing/inf/negative values', (d.isna().any(axis=1)) | (np.isinf(d.drop(columns=[])).any(axis=1)) | ((d.drop(columns=['Label']) < 0).any(axis=1)))
chk('at least 2 packets per flow', n < 2)
# deterministic-ratio probe: is any class-specific constant ratio present?
r = d.Fwd_Pkts_s / d.Flow_Pkts_s
cons = {int(k): float(g.round(6).nunique()) for k, g in r.groupby(d.Label)}
out = {'n_records': int(len(d)), 'violations': checks,
       'all_checks_pass': bool(all(v == 0 for v in checks.values())),
       'distinct_values_of_Fwd_Pkts_s/Flow_Pkts_s_by_class': cons}
json.dump(out, open(os.path.join(HERE, 'validation_results.json'), 'w'), indent=2)
for k, v in checks.items(): print(f'{k:60s} {v}')
print('ALL PASS' if out['all_checks_pass'] else 'SOME CHECKS FAILED', '| ratio distinct values:', cons)
