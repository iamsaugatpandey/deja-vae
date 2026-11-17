"""Review (CPU only): (1) attribution and bank-max TPR of the GenImage decoder bank with the JPEG q95 re-save
(runs/dv/bank_q95, from dv_bank_q95.py); (2) Deja VAE-I with vs without counterfactual rows: paired TPR@5% interval and
false alarms on degraded reals, per benchmark (Appendix "One factor at a time").
    anchor/bin/python review_cpu2.py   -> runs/review_cpu2.txt
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import anchor_pilot as ap
import dejavae as dv

L = []
O = Path("runs/dv/bank_q95"); A = dict(np.load(O / "phi_v.npz")); names = ["sd1orig", "ftmse", "kd21", "sdxl", "flux", "dcae"]
R = {n: dict(np.load(O / f"phi_R_{n}.npz")) for n in names}
def dist(a, b): return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)
def z(n, g):
    d = np.log(dist(A[g], R[n][g]) + 1e-6); r = np.log(dist(A["gi:real"], R[n]["gi:real"]) + 1e-6)
    return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)
att = {}
for g in [k for k in A if k != "gi:real"]:
    Z = np.stack([z(n, g) for n in names], 1)
    att[g[3:]] = pd.Series(np.array(names)[Z.argmax(1)]).value_counts(normalize=True).reindex(names).fillna(0) * 100
    zm = Z.max(1); rm = np.stack([z(n, "gi:real") for n in names], 1).max(1)
    att[g[3:]]["BANKmax_TPR"] = 100 * (zm > np.quantile(rm, .95)).mean()
L += ["q95 GenImage attribution (%) and bank-max TPR", pd.DataFrame(att).T.round(1).to_string()]
sd1 = np.concatenate([np.stack([z(n, g) for n in names], 1) for g in ("gi:stable_diffusion_v_1_4", "gi:stable_diffusion_v_1_5", "gi:wukong")])
thr = {n: np.quantile(z(n, "gi:real"), .95) for n in names}
L.append("pooled SD1-based (600): sd1orig TPR %.1f  ftmse TPR %.1f" % (100 * (sd1[:, 0] > thr["sd1orig"]).mean(), 100 * (sd1[:, 1] > thr["ftmse"]).mean()))
for name in ("genimage", "cham", "sb"):
    rows, idx = dv.rows_of(name); ix = np.array(idx); S = dv.scores_for(name, rows, idx); Rr = [rows[i] for i in idx]
    a = [S["dv-single"][s][0][ix] for s in (0, 1, 2)]; b = [S["dv-single-nocf"][s][0][ix] for s in (0, 1, 2)]
    m, lo, hi = ap.paired_bootstrap(a, b, Rr, eps=0.05, B=1000)["mean"]
    fd = [ap.evaluate(S[v][s][0][ix], S[v][s][1][ix], Rr)[1]["fpr_degraded_reals@5%"] for v in ("dv-single", "dv-single-nocf") for s in (0, 1, 2)]
    L.append(f"{name}: dvI(with CF) - dvI(no CF) TPR@5 {m:+.1f} [{lo:+.1f},{hi:+.1f}]  FPR_deg with CF {np.mean(fd[:3]):.1f}  without {np.mean(fd[3:]):.1f}")
txt = "\n".join(L); print(txt); Path("runs/review_cpu2.txt").write_text(txt + "\n")
