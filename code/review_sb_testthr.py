"""Review: is the final-layer Deja VAE-I lead on Synthbuster a looser threshold? Its FPR on the matched test reals is
7.4% / 6.4% (downscaled / cropped) against SimLBR's 5.6% / 5.3%. Here every method's threshold is the 95th percentile
of the matched TEST reals (equal FPR by construction), and AUROC (test reals vs each generator's images, averaged over
generators). Paired differences resample RAISE photographs (calibration rows are not used), B=1000.
    anchor/bin/python review_sb_testthr.py   -> runs/sb/testthr.txt
"""
import os, sys
os.environ["DV_FEAT"] = "mid"; os.environ["DV_WORK"] = "runs/dv2"
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv
from dv_bank_full import auroc


def stats(x, te, fk_by_g):
    t = np.quantile(x[te], 0.95)
    return np.mean([(x[f] > t).mean() * 100 for f in fk_by_g]), np.mean([auroc(x[f], x[te]) for f in fk_by_g])


L = [__doc__.strip().splitlines()[0], ""]
for SET in ("sb", "sb_crop"):
    dv.W, dv.FEAT = Path("runs/dv2"), "mid"
    rows, idx = dv.rows_of(SET); ix = np.array(idx); R = [rows[i] for i in idx]
    M = {"Deja VAE-I (layers 8-18)": dv.scores_for(SET, rows, idx)["dv-single"]}
    dv.W, dv.FEAT = Path("runs/dv"), "cls24"
    M["Deja VAE-I (final layer)"] = dv.scores_for(SET, rows, idx)["dv-single"]
    ref = Path(dv.SETS[SET][1])
    M["SimLBR (released)"] = {s: (np.load(ref / f"scores_simlbr_official_seed{s}.npy"), None) for s in (0, 1, 2)}
    M = {k: [v[s][0][ix].astype(np.float64) for s in (0, 1, 2)] for k, v in M.items()}
    lab, split, src = ap.split_idx(R); pid = np.array([Path(r["path"]).stem for r in R])
    gens = sorted(set(src[lab == 1])); te = np.where((split == "test") & (lab == 0))[0]
    fk = {g: np.where((lab == 1) & (src == g))[0] for g in gens}
    L.append(f"[{SET}] threshold = 95th percentile of the {len(te)} matched test reals")
    for k, xs in M.items():
        r = np.array([stats(x, te, list(fk.values())) for x in xs]).mean(0)
        L.append(f"  {k:26s} TPR@5 {r[0]:5.1f}   AUROC {r[1]:.3f}")
    # photo-clustered paired bootstrap, final layer minus SimLBR
    by = lambda ix_: {p: ix_[pid[ix_] == p] for p in np.unique(pid[ix_])}
    T, Fk = list(by(te).values()), list(by(np.where(lab == 1)[0]).values())
    rng = np.random.default_rng(0); D = []
    for _ in range(1000):
        t_ = np.concatenate([T[j] for j in rng.integers(0, len(T), len(T))])
        f_ = np.concatenate([Fk[j] for j in rng.integers(0, len(Fk), len(Fk))])
        fg = [f_[src[f_] == g] for g in gens]
        d = [np.subtract(stats(a, t_, fg), stats(b, t_, fg)) for a, b in zip(M["Deja VAE-I (final layer)"], M["SimLBR (released)"])]
        D.append(np.mean(d, 0))
    D = np.array(D); lo, hi = np.percentile(D, [2.5, 97.5], 0)
    L.append(f"  final layer - SimLBR: TPR {D[:, 0].mean():+.1f} [{lo[0]:+.1f},{hi[0]:+.1f}]   AUROC {D[:, 1].mean():+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]")
    L.append(""); print("\n".join(L[-6:]), flush=True)
Path("runs/sb/testthr.txt").write_text("\n".join(L) + "\n")
