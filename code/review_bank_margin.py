"""Review: margin of the exact decoder over the best OTHER autoencoder in the Synthbuster decoder bank, under both
matchings of the real photographs (downscaled: dv_bank_sb.py; cropped: dv_bank_sb_crop.py). Training-free score as in
the bank (z-scored class-token movement, layers 10-18). AUROC and TPR at the in-sample 5% threshold; 95% intervals from
B=1000 draws resampling the 200 generated and 200 real images; the competitor is re-chosen in every draw (conservative).
    DV_FEAT=mid anchor/bin/python review_bank_margin.py   -> runs/sb/bank_margin.txt
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import dv_bank_sb as B
from dv_bank_full import auroc

EXACT = {"stable-diffusion-1-3": "sd1orig", "stable-diffusion-1-4": "sd1orig", "stable-diffusion-2": "ftmse",
         "stable-diffusion-xl": "sdxl10"}
A0, R0, names = B.load()
OUT = Path("runs/sb/bank_crop")
names = [n for n in names if (OUT / f"phi_R_{n}.npz").exists()]
Ac = dict(np.load(OUT / "phi_v.npz")); Rc = {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in names}
A = {**A0, **{k.replace(":real", ":crop"): v for k, v in Ac.items()}}
R = {n: {**R0[n], **{k.replace(":real", ":crop"): v for k, v in Rc[n].items()}} for n in names}
rng = np.random.default_rng(0)
L = [__doc__.strip().splitlines()[0], f"autoencoders: {', '.join(names)}", ""]
for tag, sfx in (("downscaled reals", "real"), ("cropped reals", "crop")):
    L.append(f"[{tag}]")
    for g, ex in EXACT.items():
        ref = f"{g}:{sfx}"
        Sf = {k: B.zscore(A, R, k, f"{g}:fake", ref) for k in names}
        Sr = {k: B.zscore(A, R, k, ref, ref) for k in names}
        tpr = lambda k, f, r: 100 * (Sf[k][f] > np.quantile(Sr[k][r], 0.95)).mean()
        others = [k for k in names if k != ex]
        f0, r0 = np.arange(len(Sf[ex])), np.arange(len(Sr[ex]))
        def margins(f, r):
            au = {k: auroc(Sf[k][f], Sr[k][r]) for k in names}; tp = {k: tpr(k, f, r) for k in names}
            ka, kt = max(others, key=au.get), max(others, key=tp.get)
            return au[ex] - au[ka], tp[ex] - tp[kt], ka, kt
        m_au, m_tp, ka, kt = margins(f0, r0)
        D = np.array([margins(rng.choice(f0, len(f0)), rng.choice(r0, len(r0)))[:2] for _ in range(1000)])
        lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
        L.append(f"  {g:22s} exact={ex:7s} AUROC margin {m_au:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}] (vs {ka})   "
                 f"TPR margin {m_tp:+5.1f} [{lo[1]:+.1f},{hi[1]:+.1f}] (vs {kt})")
        print(L[-1], flush=True)
    L.append("")
Path("runs/sb/bank_margin.txt").write_text("\n".join(L) + "\n")
