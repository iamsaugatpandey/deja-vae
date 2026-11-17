"""E5 extra paired comparisons on Synthbuster (not in dejavae report): Deja VAE-I vs every baseline, fusion controls,
and per-generator paired differences. Same protocol as anchor_pilot.paired_bootstrap (resample calibration reals and
each generator's test fakes; seed-averaged; B=1000). Output: runs/sb/extra.txt
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python sb_extra.py [sb|sb_crop]   (sb_crop: crop-matched reals -> runs/sb_crop/extra.txt)
"""
import os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent / "figures"))
import numpy as np
import anchor_pilot as ap
import dejavae as dv
import fig_fusion as ff

SET = sys.argv[1] if len(sys.argv) > 1 else "sb"
rows, runs = ff.runs_for(SET)
_, idx = dv.rows_of(SET); ix = np.array(idx); ref = Path(dv.SETS[SET][1])
for m in [m for m in ("rigid-dinov3", "lbr_exact") if (ref / f"scores_{m}_seed0.npy").exists()]:
    runs[m] = {s: (np.load(ref / f"scores_{m}_seed{s}.npy")[ix], None) for s in (0, 1, 2)}
S = lambda m: [runs[m][s][0] for s in (0, 1, 2)]
L = [f"E5 extra paired comparisons, {SET} (TPR@5%, seed-averaged, B=1000)", ""]
for a, b in [("dv-single", "simlbr_official"), ("dv-single", "lbr_exact"), ("dv-single", "sup"), ("dv-single", "recon"),
             ("dv-single", "rigid-dinov3"), ("dv-single", "dv-tf"),
             ("dv-single+fuse-simlbr_official", "simlbr_official"),
             ("dv-single+fuse-simlbr_official", "sup+fuse-simlbr_official"),
             ("dv-single+fuse-simlbr_official", "recon+fuse-simlbr_official"),
             ("dv-single+fuse-simlbr_official", "rigid-dinov2+fuse-simlbr_official"),
             ("sup+fuse-simlbr_official", "simlbr_official"), ("recon+fuse-simlbr_official", "simlbr_official")]:
    if a not in runs or b not in runs:
        continue
    r = ap.paired_bootstrap(S(a), S(b), rows, eps=0.05, B=1000)
    (mm, ml, mh), (wm, wl, wh) = r["mean"], r["worst"]
    L.append(f"  {a:32s} - {b:34s} mean {mm:+5.1f} [{ml:+.1f},{mh:+.1f}]  worst {wm:+5.1f} [{wl:+.1f},{wh:+.1f}]")
    print(L[-1], flush=True)

# per-generator paired differences
lab, split, src = ap.split_idx(rows); rng = np.random.default_rng(1); B = 1000
cal = np.where((split == "calib") & (lab == 0))[0]; gens = sorted(set(src[(split == "test") & (lab == 1)]))
fk = {g: np.where((split == "test") & (src == g) & (lab == 1))[0] for g in gens}
L += ["", "Per-generator paired difference in TPR@5% (pooled threshold; bootstrap over calib reals and that generator's fakes)"]
for a, b in [("dv-single", "simlbr_official"), ("dv", "dv-single"), ("dv-single+fuse-simlbr_official", "simlbr_official")]:
    L.append(f"  {a} - {b}")
    for g in gens:
        d = np.zeros(B + 1)
        for k in range(B + 1):
            c = cal if k == 0 else rng.choice(cal, len(cal)); f = fk[g] if k == 0 else rng.choice(fk[g], len(fk[g]))
            v = []
            for xa, xb in zip(S(a), S(b)):
                v.append(((xa[f] > ap.conformal_tau(xa[c], .05)).mean() - (xb[f] > ap.conformal_tau(xb[c], .05)).mean()) * 100)
            d[k] = np.mean(v)
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        L.append(f"    {g:24s} {d[0]:+6.1f} [{lo:+.1f},{hi:+.1f}]"); print(L[-1], flush=True)
(ref / "extra.txt").write_text("\n".join(L) + "\n")
