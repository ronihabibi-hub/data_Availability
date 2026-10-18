"""
01_generate_dataset.py  (version 3, Round-3 revision)

Synthetic flow-level dataset for a campus (academic-network) flood-detection
scenario.  Version 3 replaces the version-2 generator, which drew each
feature independently inside a "normal" block and a "flood" block and tied
several features together with fixed class-specific constants
(Fwd_Pkts_s = 0.55 x Flow_Pkts_s for normal, 0.85 x for flood).

Principle of this version: every flow is first simulated at the PACKET level
(packet counts per direction, per-packet payload lengths, inter-arrival
gaps, header size, TCP flag events).  All 25 reported flow features are then
COMPUTED from those underlying quantities, as a flow meter would, so
min <= mean <= max, byte totals, durations, rates and directional ratios are
consistent by construction.  No class-specific constant relates one feature
to another.  The label is the generating class (benign profile vs flood
variant); it is never a threshold on any feature.

Benign traffic is a mixture of application profiles typical of a campus
network (web/LMS browsing, lecture-video streaming, video conferencing, DNS,
interactive SSH to lab machines, bulk transfers in both directions, LMS
flash-crowd bursts around deadlines/exams, and chat/mail).  Flood traffic is
a mixture of SYN, UDP and HTTP floods, a fraction of which is generated at
low rate ("stealth" floods) so that attack and benign flows overlap in the
feature space.  Profile weights and parameters are modelling ASSUMPTIONS
informed by published campus-traffic observations (see the manuscript); they
are not fitted to a real packet capture.

Usage:  python 01_generate_dataset.py [--n 2500] [--stealth 0.25]
                                      [--mix baseline|lecture|exam|lab]
                                      [--seed 77] [--out network_flood_dataset.csv]
"""
import argparse
import os
import numpy as np
import pandas as pd

SEED = 77
HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    'Flow_Duration', 'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts',
    'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min',
    'Flow_Byts_s', 'Flow_Pkts_s', 'Flow_IAT_Mean', 'Flow_IAT_Max', 'Flow_IAT_Min',
    'Fwd_Header_Len', 'Bwd_Header_Len', 'Fwd_Pkts_s', 'Bwd_Pkts_s',
    'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt', 'ACK_Flag_Cnt', 'URG_Flag_Cnt', 'Down_Up_Ratio']

# ---- payload-length components (bytes of payload, TCP MSS 1448-1460) -------
def draw_len(rng, k, mix):
    """mix = (p_zero, p_small, p_medium, p_large)"""
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    c = rng.choice(4, size=k, p=np.asarray(mix) / np.sum(mix))
    out = np.zeros(k, dtype=np.int64)
    m = c == 1; out[m] = rng.integers(1, 121, m.sum())
    m = c == 2; out[m] = rng.integers(121, 801, m.sum())
    m = c == 3; out[m] = rng.integers(1448, 1461, m.sum())
    return out

def lognint(rng, median, sigma, lo, hi):
    return int(np.clip(round(rng.lognormal(np.log(median), sigma)), lo, hi))

# ---- benign application profiles ------------------------------------------
# each: proto, counts(rng)->(nf,nb), fwd/bwd length mixes, gap (median_us, sigma),
#       psh prob, rst prob
def p_web(rng):
    nf = lognint(rng, 12, 0.9, 2, 400); nb = max(1, int(round(nf * rng.lognormal(np.log(1.6), 0.5)))); return nf, nb
def p_video(rng):
    nf = lognint(rng, 120, 0.8, 5, 1500); nb = max(1, int(round(nf * rng.lognormal(np.log(8), 0.4)))); return nf, min(nb, 3000)
def p_conf(rng):
    nf = lognint(rng, 300, 0.8, 10, 2500); nb = max(1, int(round(nf * rng.lognormal(np.log(1.0), 0.25)))); return nf, nb
def p_dns(rng):
    return (1 if rng.random() < 0.93 else 2), 1
def p_ssh(rng):
    nf = lognint(rng, 60, 1.0, 3, 1500); nb = max(1, int(round(nf * rng.lognormal(np.log(1.1), 0.4)))); return nf, nb
def p_bulk(rng):
    if rng.random() < 0.6:   # download from mirror / repository
        nf = lognint(rng, 80, 0.9, 5, 1500); nb = max(1, int(round(nf * rng.lognormal(np.log(2.0), 0.3))))
    else:                    # upload (backup, dataset, assignment submission)
        nf = lognint(rng, 400, 0.7, 20, 3000); nb = max(1, int(round(nf * rng.lognormal(np.log(0.5), 0.3))))
    return nf, nb
def p_lms(rng):
    nf = lognint(rng, 5, 0.6, 2, 60)
    nb = 0 if rng.random() < 0.12 else max(0, int(round(nf * rng.lognormal(np.log(0.6), 0.6))))
    return nf, nb
def p_chat(rng):
    nf = lognint(rng, 8, 0.8, 2, 300); nb = max(1, int(round(nf * rng.lognormal(np.log(1.0), 0.4)))); return nf, nb

PROFILES = {
 #            proto  counts  fwd_len_mix            bwd_len_mix             gap(med us,sigma) psh   rst
 'web':      ('tcp', p_web,   (.55,.10,.33,.02),     (.30,.10,.20,.40),      (8000, 1.6),      .25,  .05),
 'video':    ('tcp', p_video, (.85,.08,.06,.01),     (.05,.02,.08,.85),      (3000, 1.2),      .08,  .02),
 'conf':     ('udp', p_conf,  (.00,.25,.60,.15),     (.00,.25,.60,.15),      (15000, .5),      .00,  .00),
 'dns':      ('udp', p_dns,   (.00,1.0,.00,.00),     (.00,.35,.65,.00),      (20000, .9),      .00,  .00),
 'ssh':      ('tcp', p_ssh,   (.45,.50,.05,.00),     (.30,.45,.20,.05),      (150000, 1.5),    .60,  .03),
 'bulk':     ('tcp', p_bulk,  None,                  None,                   (150, 1.0),       .03,  .03),
 'lms':      ('tcp', p_lms,   (.50,.10,.38,.02),     (.35,.20,.25,.20),      (2000, 1.3),      .30,  .12),
 'chat':     ('tcp', p_chat,  (.40,.30,.25,.05),     (.40,.30,.25,.05),      (40000, 1.5),     .30,  .05),
}
MIXES = {   # benign application weights (assumptions; see manuscript)
 'baseline': dict(web=.34, video=.10, conf=.07, dns=.20, ssh=.04, bulk=.07, lms=.10, chat=.08),
 'lecture':  dict(web=.28, video=.20, conf=.15, dns=.17, ssh=.02, bulk=.05, lms=.05, chat=.08),
 'exam':     dict(web=.30, video=.04, conf=.03, dns=.18, ssh=.02, bulk=.03, lms=.32, chat=.08),
 'lab':      dict(web=.25, video=.06, conf=.03, dns=.18, ssh=.16, bulk=.18, lms=.05, chat=.09),
}
ATTACKS = {'syn': .35, 'udp': .30, 'http': .35}

def tcp_hdr(rng, attack):
    """per-endpoint TCP header size: 32 (timestamps option) or 20; stack-dependent."""
    p32 = 0.30 if attack else 0.55
    return 32 if rng.random() < p32 else 20

def sample_gaps(rng, k, med, sigma, extra_pause=0.0):
    if k <= 0:
        return np.zeros(0)
    g = rng.lognormal(np.log(med), sigma, k)
    if extra_pause > 0:
        m = rng.random(k) < extra_pause
        g[m] = rng.lognormal(np.log(2e6), 0.5, m.sum())
    return np.maximum(g, 1.0)           # >= 1 microsecond timestamp resolution

def finish(rng, nf, nb, fl, bl, gaps, hf, hb, syn, rst, psh, ack, urg):
    n = nf + nb
    D = float(np.sum(gaps)) if len(gaps) else 1.0
    D = max(D, 1.0)
    totf, totb = int(fl.sum()), int(bl.sum())
    sec = D / 1e6
    rec = {
     'Flow_Duration': D, 'Tot_Fwd_Pkts': nf, 'Tot_Bwd_Pkts': nb,
     'TotLen_Fwd_Pkts': totf, 'TotLen_Bwd_Pkts': totb,
     'Fwd_Pkt_Len_Max': int(fl.max()) if nf else 0, 'Fwd_Pkt_Len_Min': int(fl.min()) if nf else 0,
     'Fwd_Pkt_Len_Mean': float(fl.mean()) if nf else 0.0,
     'Bwd_Pkt_Len_Max': int(bl.max()) if nb else 0, 'Bwd_Pkt_Len_Min': int(bl.min()) if nb else 0,
     'Flow_Byts_s': (totf + totb) / sec, 'Flow_Pkts_s': n / sec,
     'Flow_IAT_Mean': float(gaps.mean()), 'Flow_IAT_Max': float(gaps.max()), 'Flow_IAT_Min': float(gaps.min()),
     'Fwd_Header_Len': nf * hf, 'Bwd_Header_Len': nb * hb,
     'Fwd_Pkts_s': nf / sec, 'Bwd_Pkts_s': nb / sec,
     'SYN_Flag_Cnt': int(min(syn, n)), 'RST_Flag_Cnt': int(min(rst, n)), 'PSH_Flag_Cnt': int(min(psh, n)),
     'ACK_Flag_Cnt': int(min(ack, n)), 'URG_Flag_Cnt': int(urg),
     'Down_Up_Ratio': nb / nf}
    return rec

def benign_flow(rng, name):
    proto, cnt, fm, bm, (gmed, gsig), ppsh, prst = PROFILES[name]
    nf, nb = cnt(rng)
    nf = max(nf, 1)
    if nf + nb < 2:
        nb = 1
    if name == 'bulk':
        upload = nf > nb
        if upload:   # data flows forward
            fm, bm = (.05, .05, .02, .88), (.92, .08, .0, .0)
        else:
            fm, bm = (.92, .08, .0, .0), (.04, .02, .02, .92)
    fl = draw_len(rng, nf, fm); bl = draw_len(rng, nb, bm)
    n = nf + nb
    gaps = sample_gaps(rng, n - 1, gmed * rng.lognormal(0, 0.35), gsig, 0.02 if name == 'video' else 0.0)
    if proto == 'tcp':
        hf, hb = tcp_hdr(rng, False), tcp_hdr(rng, False)
        has_syn = rng.random() < 0.90
        syn = (1 + (1 if nb > 0 else 0) + (1 if rng.random() < 0.04 else 0)) if has_syn else 0
        ack = max(0, n - (1 if has_syn else 0) - rng.binomial(n, 0.02))
        psh = rng.binomial(n, ppsh)
        rst = (1 + rng.poisson(0.3)) if rng.random() < prst else 0
    else:
        hf = hb = 8; syn = rst = psh = ack = 0
    urg = 1 if (proto == 'tcp' and rng.random() < 0.003) else 0
    return finish(rng, nf, nb, fl, bl, gaps, hf, hb, syn, rst, psh, ack, urg)

def flood_flow(rng, kind, stealth_frac):
    stealth = rng.random() < stealth_frac
    if kind == 'syn':
        nf = 2 + lognint(rng, 6, 0.9, 0, 150)
        nb = rng.binomial(nf, rng.uniform(0.0, 0.6))
        tool_payload = rng.random() < 0.3
        fm = (.2, .6, .2, .0) if tool_payload else (1.0, 0, 0, 0)
        fl = draw_len(rng, nf, fm); bl = np.zeros(nb, dtype=np.int64)
        lo, hi = (3000, 50000) if stealth else (100, 1000)
        proto = 'tcp'
    elif kind == 'udp':
        nf = 2 + lognint(rng, 30, 1.0, 0, 800)
        nb = 0 if rng.random() < 0.7 else int(rng.integers(1, 6))
        r = rng.random()
        if r < 0.25:   fl = rng.integers(8, 1473, nf)                              # random payload size
        else:
            base = 60 if r < 0.45 else (512 if r < 0.65 else 1400)
            fl = np.clip(np.round(base * (1 + rng.normal(0, 0.03, nf))), 0, 1472).astype(np.int64)
        bl = draw_len(rng, nb, (0, .8, .2, 0))
        lo, hi = (2000, 30000) if stealth else (50, 500)
        proto = 'udp'
    else:  # http flood over established connections
        nf = 2 + lognint(rng, 15, 0.8, 0, 300)
        nb = int(round(nf * rng.uniform(0.2, 1.2)))
        fl = draw_len(rng, nf, (.10, .15, .75, .0)); bl = draw_len(rng, nb, (.30, .30, .20, .20))
        lo, hi = (5000, 60000) if stealth else (300, 3000)
        proto = 'tcp'
    n = nf + nb
    gmed = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    gaps = sample_gaps(rng, n - 1, gmed, 0.9)
    if proto == 'tcp':
        hf, hb = tcp_hdr(rng, True), tcp_hdr(rng, True)
        if kind == 'syn':
            syn = rng.binomial(nf, 0.95) + rng.binomial(nb, 0.9)
            ack = rng.binomial(nb, 0.9) + rng.binomial(nf, 0.05)
            psh = rng.binomial(n, 0.01)
            rst = (1 + rng.poisson(1.0)) if rng.random() < 0.3 else 0
        else:
            syn = 1 + (1 if nb > 0 else 0)
            ack = max(0, n - 1 - rng.binomial(n, 0.02))
            psh = rng.binomial(nf, 0.8) + rng.binomial(nb, 0.15)
            rst = (1 + rng.poisson(0.3)) if rng.random() < 0.2 else 0
    else:
        hf = hb = 8; syn = rst = psh = ack = 0
    urg = 1 if (proto == 'tcp' and rng.random() < 0.003) else 0
    return finish(rng, nf, nb, fl, bl, gaps, hf, hb, syn, rst, psh, ack, urg), stealth

def generate(seed=SEED, n_samples=2500, stealth_frac=0.25, mix='baseline', flood_share=0.20):
    rng = np.random.default_rng(seed)
    n_flood = int(round(n_samples * flood_share)); n_norm = n_samples - n_flood
    rows, meta = [], []
    names = list(MIXES[mix].keys()); w = np.array([MIXES[mix][k] for k in names]); w = w / w.sum()
    for prof in rng.choice(names, size=n_norm, p=w):
        r = benign_flow(rng, prof); r['Label'] = 0; rows.append(r)
        meta.append(dict(Label=0, Profile=prof, Stealth=0))
    kinds = list(ATTACKS.keys()); kp = np.array(list(ATTACKS.values()))
    for kind in rng.choice(kinds, size=n_flood, p=kp):
        r, st = flood_flow(rng, kind, stealth_frac); r['Label'] = 1; rows.append(r)
        meta.append(dict(Label=1, Profile=kind + '_flood', Stealth=int(st)))
    df = pd.DataFrame(rows)[FEATURES + ['Label']]
    md = pd.DataFrame(meta)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), md.iloc[perm].reset_index(drop=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2500); ap.add_argument('--stealth', type=float, default=0.25)
    ap.add_argument('--mix', default='baseline'); ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(HERE, 'network_flood_dataset.csv'))
    a = ap.parse_args()
    df, md = generate(a.seed, a.n, a.stealth, a.mix)
    df.to_csv(a.out, index=False)
    md.to_csv(a.out.replace('.csv', '_metadata.csv'), index=False)
    print(f'Generated {len(df)} flow records -> {a.out}'); print(df['Label'].value_counts())
