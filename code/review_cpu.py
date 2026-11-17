"""Review analyses on cached scores (CPU): (A) Deja VAE-I/P layer ablation (final layer = v1 runs/dv vs layers 8-18 = v2);
(B) Table 1 key numbers with the 470+350 layer-pilot images (dv_layers.groups, seed 7) removed from the evaluation."""
import os, sys, json
os.environ.setdefault("DV_FEAT", "mid")
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import anchor_pilot as ap, dejavae as dv, dv_layers as DL
L = []
for name, ref in (("genimage", "runs/p4s"), ("cham", "runs/cham_off")):
    rows, idx = dv.rows_of(name); ix = np.array(idx); R = [rows[i] for i in idx]
    v1 = dict(np.load(f"runs/dv/v1_scores_{name}.npz")); v2 = dict(np.load(f"runs/dv2/e8_scores_{name}.npz"))
    for v, pub in (("dv-single", "Deja VAE-I"), ("dv", "Deja VAE-P")):
        r = ap.paired_bootstrap([v2[f"{v}|{s}"] for s in (0, 1, 2)], [v1[f"{v}|{s}"] for s in (0, 1, 2)], R, eps=0.05, B=1000)
        (m, lo, hi), (w, wl, wh) = r["mean"], r["worst"]
        L.append(f"(A) {name:9s} {pub}: layers 8-18 minus final layer  mean {m:+.1f} [{lo:+.1f},{hi:+.1f}]  worst {w:+.1f} [{wl:+.1f},{wh:+.1f}]")
    # (B) pilot exclusion
    G = DL.groups(); pil = {p for k, ps in G.items() if k.startswith("gi:" if name == "genimage" else "cham:") for p in ps}
    keep = np.array([r["path"] not in pil for r in R]); Rk = [r for r, k in zip(R, keep) if k]
    sl = np.load(f"{ref}/scores_simlbr_official_seed0.npy")[ix]
    runs = {"SimLBR (released)": [sl] * 3, "Deja VAE-I": [v2[f"dv-single|{s}"] for s in (0, 1, 2)],
            "Deja VAE-P": [v2[f"dv|{s}"] for s in (0, 1, 2)]}
    for k, xs in runs.items():
        full = np.mean([ap.evaluate(x, x, R)[1]["mean_tpr@5%"] for x in xs])
        red = np.mean([ap.evaluate(x[keep], x[keep], Rk)[1]["mean_tpr@5%"] for x in xs])
        L.append(f"(B) {name:9s} {k:18s} TPR@5 all {full:5.1f}  without pilot images {red:5.1f}  ({(~keep).sum()} rows removed)")
print("\n".join(L)); Path("runs/review_cpu.txt").write_text("\n".join(L) + "\n")
