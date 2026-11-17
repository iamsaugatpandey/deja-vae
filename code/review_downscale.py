"""Review (CPU only): on Synthbuster's downscale-matched TEST reals, is the score related to how strongly the RAISE
photograph was downscaled? Spearman correlation of the downscale factor with Deja VAE-I (seed mean) and SimLBR scores.
    DV_FEAT=mid DV_WORK=runs/dv2 anchor/bin/python review_downscale.py [RAISE_TIFF_DIR]   -> runs/review_downscale.txt
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import anchor_pilot as ap
import dejavae as dv

T = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "anchor_data/raise/tiff"
rows, idx = dv.rows_of("sb"); ix = np.array(idx); S = dv.scores_for("sb", rows, idx)
x = np.mean([S["dv-single"][s][0] for s in (0, 1, 2)], 0); sl = np.load("runs/sb/scores_simlbr_official_seed0.npy")
lab, split, src = ap.split_idx(rows); W = {}; fac = np.full(len(rows), np.nan)
for i in ix:
    r = rows[i]
    if r["label"] != 0:
        continue
    rid = Path(r["path"]).stem
    if rid not in W:
        with Image.open(T / f"{rid}.TIF") as im:
            W[rid] = im.size
    with Image.open(r["path"]) as im:
        w, h = im.size
    fac[i] = min(W[rid][0] / w, W[rid][1] / h)      # downscale factor of the matched real (limiting side)
m = (lab == 0) & (split == "test") & ~np.isnan(fac)
L = ["Matched TEST reals: Spearman(downscale factor, score), all generators pooled",
     f"  Deja VAE-I  rho={spearmanr(fac[m], x[m])[0]:+.3f}   SimLBR rho={spearmanr(fac[m], sl[m])[0]:+.3f}   n={m.sum()}",
     "Per generator: median downscale factor, mean Deja VAE-I score on matched reals (higher = more 'generated')"]
for g in sorted(set(src[m])):
    k = m & (src == g)
    L.append(f"  {g:22s} factor {np.median(fac[k]):5.1f}  dvI mean {x[k].mean():+.2f}  within-gen rho {spearmanr(fac[k], x[k])[0]:+.2f}")
txt = "\n".join(L); print(txt); Path("runs/review_downscale.txt").write_text(txt + "\n")
