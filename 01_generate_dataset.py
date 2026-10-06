"""
01_generate_dataset.py

Generates the synthetic flow-based network-traffic dataset used throughout
this study. This is a verbatim, fully-documented reproduction of the
generation procedure originally implemented by the author in Google Colab.

Design summary (for the manuscript's Methods section):
- 2,500 total records: 2,000 "normal" (80%) and 500 "flood" (20%).
- 25 flow-based features, generated independently as two blocks (normal,
  flood) using different distribution parameters per block, then
  concatenated and shuffled. The label is therefore assigned BY
  CONSTRUCTION (which generation block a record came from), not by any
  post-hoc thresholding rule on the feature values.
- Flood-block parameters are deliberately set to emulate a high-volume,
  low-diversity flooding pattern: shorter flow durations, higher forward
  packet counts with fewer backward packets (asymmetric, upload-heavy),
  lower Down_Up_Ratio, and shorter inter-arrival times (IAT) than the
  normal-traffic block. This is consistent with the generic signature of
  volumetric flood/DDoS traffic (e.g., SYN flood, UDP flood, HTTP flood)
  described in the flow-based IDS literature, rather than a single named
  attack tool or protocol.
- A fixed random seed (77) is used for every distribution draw and for the
  final shuffle, so the dataset is exactly reproducible from this script.
"""

import pandas as pd
import numpy as np
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "network_flood_dataset.csv")
SEED = 77

def generate(seed=SEED, n_samples=2500):
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Normal traffic block (80% of records, n = 2000)
    # ------------------------------------------------------------------
    n_normal = int(n_samples * 0.8)
    df_normal = pd.DataFrame({
        'Flow_Duration': np.random.exponential(scale=65000, size=n_normal) + np.random.uniform(50, 8000, n_normal),
        'Tot_Fwd_Pkts': np.random.negative_binomial(n=8, p=0.25, size=n_normal) + 1,
        'Tot_Bwd_Pkts': np.random.negative_binomial(n=6, p=0.25, size=n_normal) + 1,
    })
    df_normal['TotLen_Fwd_Pkts'] = df_normal['Tot_Fwd_Pkts'] * np.random.uniform(40, 1200, n_normal)
    df_normal['TotLen_Bwd_Pkts'] = df_normal['Tot_Bwd_Pkts'] * np.random.uniform(40, 1400, n_normal)
    df_normal['Fwd_Pkt_Len_Max'] = np.random.uniform(64, 1500, n_normal)
    df_normal['Fwd_Pkt_Len_Min'] = np.random.choice([0, 64, 100], size=n_normal, p=[0.5, 0.4, 0.1])
    df_normal['Fwd_Pkt_Len_Mean'] = np.random.uniform(60, 600, n_normal)
    df_normal['Bwd_Pkt_Len_Max'] = np.random.uniform(64, 1500, n_normal)
    df_normal['Bwd_Pkt_Len_Min'] = np.random.choice([0, 64, 100], size=n_normal, p=[0.4, 0.4, 0.2])
    df_normal['Flow_Byts_s'] = np.random.lognormal(mean=11.2, sigma=1.6, size=n_normal)
    df_normal['Flow_Pkts_s'] = np.random.lognormal(mean=6.2, sigma=1.4, size=n_normal)
    df_normal['Flow_IAT_Mean'] = np.random.uniform(10, 800, n_normal)
    df_normal['Flow_IAT_Max'] = np.random.uniform(100, 8000, n_normal)
    df_normal['Flow_IAT_Min'] = np.random.uniform(0.1, 25, n_normal)
    df_normal['Fwd_Header_Len'] = df_normal['Tot_Fwd_Pkts'] * 20
    df_normal['Bwd_Header_Len'] = df_normal['Tot_Bwd_Pkts'] * 20
    df_normal['Fwd_Pkts_s'] = df_normal['Flow_Pkts_s'] * 0.55
    df_normal['Bwd_Pkts_s'] = df_normal['Flow_Pkts_s'] * 0.45
    df_normal['SYN_Flag_Cnt'] = np.random.choice([0, 1], n_normal, p=[0.75, 0.25])
    df_normal['RST_Flag_Cnt'] = np.random.choice([0, 1], n_normal, p=[0.95, 0.05])
    df_normal['PSH_Flag_Cnt'] = np.random.choice([0, 1], n_normal, p=[0.45, 0.55])
    df_normal['ACK_Flag_Cnt'] = np.random.choice([0, 1], n_normal, p=[0.35, 0.65])
    df_normal['URG_Flag_Cnt'] = np.random.choice([0, 1], n_normal, p=[0.99, 0.01])
    df_normal['Down_Up_Ratio'] = np.random.uniform(0.4, 2.5, n_normal)
    df_normal['Label'] = 0

    # ------------------------------------------------------------------
    # Flood traffic block (20% of records, n = 500)
    # ------------------------------------------------------------------
    n_flood = int(n_samples * 0.2)
    df_flood = pd.DataFrame({
        'Flow_Duration': np.random.exponential(scale=20000, size=n_flood) + np.random.uniform(20, 3000, n_flood),
        'Tot_Fwd_Pkts': np.random.negative_binomial(n=12, p=0.15, size=n_flood) + 4,
        'Tot_Bwd_Pkts': np.random.randint(0, 20, size=n_flood),
    })
    df_flood['TotLen_Fwd_Pkts'] = df_flood['Tot_Fwd_Pkts'] * np.random.uniform(40, 450, n_flood)
    df_flood['TotLen_Bwd_Pkts'] = df_flood['Tot_Bwd_Pkts'] * np.random.uniform(0, 450, n_flood)
    df_flood['Fwd_Pkt_Len_Max'] = np.random.uniform(40, 1100, n_flood)
    df_flood['Fwd_Pkt_Len_Min'] = np.random.choice([0, 64, 100], size=n_flood, p=[0.3, 0.6, 0.1])
    df_flood['Fwd_Pkt_Len_Mean'] = np.random.uniform(40, 350, n_flood)
    df_flood['Bwd_Pkt_Len_Max'] = np.random.uniform(0, 1100, n_flood)
    df_flood['Bwd_Pkt_Len_Min'] = np.random.choice([0, 64], size=n_flood, p=[0.7, 0.3])
    df_flood['Flow_Byts_s'] = np.random.lognormal(mean=12.5, sigma=1.9, size=n_flood)
    df_flood['Flow_Pkts_s'] = np.random.lognormal(mean=7.8, sigma=1.6, size=n_flood)
    df_flood['Flow_IAT_Mean'] = np.random.uniform(0.5, 150, n_flood)
    df_flood['Flow_IAT_Max'] = np.random.uniform(5, 3000, n_flood)
    df_flood['Flow_IAT_Min'] = np.random.uniform(0.0, 5, n_flood)
    df_flood['Fwd_Header_Len'] = df_flood['Tot_Fwd_Pkts'] * 20
    df_flood['Bwd_Header_Len'] = df_flood['Tot_Bwd_Pkts'] * 20
    df_flood['Fwd_Pkts_s'] = df_flood['Flow_Pkts_s'] * 0.85
    df_flood['Bwd_Pkts_s'] = df_flood['Flow_Pkts_s'] * 0.15
    df_flood['SYN_Flag_Cnt'] = np.random.choice([0, 1], n_flood, p=[0.25, 0.75])
    df_flood['RST_Flag_Cnt'] = np.random.choice([0, 1], n_flood, p=[0.75, 0.25])
    df_flood['PSH_Flag_Cnt'] = np.random.choice([0, 1], n_flood, p=[0.85, 0.15])
    df_flood['ACK_Flag_Cnt'] = np.random.choice([0, 1], n_flood, p=[0.75, 0.25])
    df_flood['URG_Flag_Cnt'] = np.random.choice([0, 1], n_flood, p=[0.99, 0.01])
    df_flood['Down_Up_Ratio'] = np.random.uniform(0.0, 0.5, n_flood)
    df_flood['Label'] = 1

    df = pd.concat([df_normal, df_flood], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} records -> {OUT_PATH}")
    print(df['Label'].value_counts())
