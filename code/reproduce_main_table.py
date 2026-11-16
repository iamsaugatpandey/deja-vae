"""Recompute Table 1 of the paper from the released per-image scores alone (no images, no GPU).
    python reproduce_main_table.py ../data/scores
Needs numpy and pandas. Thresholds: split conformal on the calibration reals (k = ceil((n+1)(1-eps))-th smallest score).
TPR per generator, then mean and worst over generators; FPR on degraded test reals at the 5% threshold; mean over seeds.
Fusion: each detector's scores standardised by the mean and std of its calibration-real scores, then summed.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/scores")
METHODS = ["SimLBR_released", "SimLBR_retrained", "supervised_probe", "RIGID_dinov2_pixel", "RIGID_dinov2_norm",
           "RIGID_dinov3", "DejaVAE_TF", "reconstruction_probe", "DejaVAE_I", "DejaVAE_P", "DejaVAE_P+SimLBR_released"]


def tau(cal, eps):
    k = math.ceil((len(cal) + 1) * (1 - eps))
    return np.inf if k > len(cal) else np.sort(cal)[k - 1]


def seeds(df, m):
    cols = sorted(c for c in df.columns if c.startswith(m + "@seed") and not c.endswith("|degraded"))
    return [(df[c].to_numpy(float), df[c + "|degraded"].to_numpy(float)) for c in cols]


def fused(df, a, b):
    out = []
    cal = ((df.split == "calib") & (df.label == 0)).to_numpy()
    for (x, xd), (y, yd) in zip(seeds(df, a), seeds(df, b) * (3 if len(seeds(df, b)) == 1 else 1)):
        z = lambda s, ref: (s - ref[cal].mean()) / ref[cal].std()
        out.append((z(x, x) + z(y, y), z(xd, x) + z(yd, y)))
    return out


def evaluate(df, runs):
    lab, split, src = df.label.to_numpy(), df.split.to_numpy(), df.source.to_numpy()
    cal = (split == "calib") & (lab == 0); treal = (split == "test") & (lab == 0)
    gens = sorted(set(src[(split == "test") & (lab == 1)]))
    res = []
    for x, xd in runs:
        t5, t1 = tau(x[cal], 0.05), tau(x[cal], 0.01)
        tp5 = [100 * (x[(split == "test") & (lab == 1) & (src == g)] > t5).mean() for g in gens]
        tp1 = [100 * (x[(split == "test") & (lab == 1) & (src == g)] > t1).mean() for g in gens]
        res.append([np.mean(tp5), np.min(tp5), np.mean(tp1), 100 * (xd[treal] > t5).mean()])
    return np.mean(res, 0)


rows = []
for bench in ("genimage", "chameleon", "synthbuster_geometry_matched"):
    df = pd.read_csv(D / f"{bench}.csv.gz")
    for m in METHODS:
        runs = fused(df, "DejaVAE_P", "SimLBR_released") if m.endswith("+SimLBR_released") else seeds(df, m)
        if runs:
            r = evaluate(df, runs)
            rows.append(dict(benchmark=bench, method=m, tpr5=r[0], worst5=r[1], tpr1=r[2], fpr_deg=r[3]))
T = pd.DataFrame(rows).pivot(index="method", columns="benchmark").round(1)
pd.set_option("display.width", 250)
print(T.reindex(METHODS).to_string())
