"""E8: Deja VAE with SD1's ORIGINAL decoder (runs/dv4) vs ft-MSE (runs/dv2), paired per generator. Predictions stated before the run (paper, Sec. 3.4 and Table 4).
    DV_FEAT=mid anchor/bin/python e8_compare.py dump runs/dv2 ; ... dump runs/dv4 ; ... compare   -> runs/dv4/e8_compare.txt
dump: per set, seed-wise test scores of dv-single / dv / dv-tf from that work dir (runs/<work>/e8_scores_<set>.npz).
compare: paired bootstrap (calib reals + each generator's fakes resampled jointly, seed-averaged; B=1000), dv4 - dv2.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid")
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
SETS = ("genimage", "sb", "cham"); VARS = ("dv-single", "dv", "dv-tf")


def dump(work):
    os.environ["DV_WORK"] = work
    import dejavae as dv
    for name in SETS:
        rows, idx = dv.rows_of(name); ix = np.array(idx)
        runs = dv.scores_for(name, rows, idx)
        np.savez(Path(work) / f"e8_scores_{name}.npz", **{f"{v}|{s}": runs[v][s][0][ix] for v in VARS for s in (0, 1, 2)})
        print(work, name, "dumped", flush=True)


def compare():
    import anchor_pilot as ap, dejavae as dv
    L = ["E8: SD1-original-decoder copies (runs/dv4) minus ft-MSE copies (runs/dv2); TPR@5% (pooled calib threshold), B=1000", ""]
    rng = np.random.default_rng(0); B = 1000
    for name in SETS:
        rows, idx = dv.rows_of(name); rows = [rows[i] for i in idx]
        lab, split, src = ap.split_idx(rows)
        cal = np.where((split == "calib") & (lab == 0))[0]; gens = sorted(set(src[(split == "test") & (lab == 1)]))
        fk = {g: np.where((split == "test") & (lab == 1) & (src == g))[0] for g in gens}
        A = {w: dict(np.load(f"runs/{w}/e8_scores_{name}.npz")) for w in ("dv2", "dv4")}
        for v in VARS:
            S = {w: [A[w][f"{v}|{s}"] for s in (0, 1, 2)] for w in A}
            r = ap.paired_bootstrap(S["dv4"], S["dv2"], rows, eps=0.05, B=B)
            (mm, ml, mh), (wm, wl, wh) = r["mean"], r["worst"]
            L.append(f"{name:9s} {v:10s} MEAN  {mm:+6.1f} [{ml:+.1f},{mh:+.1f}]   worst {wm:+.1f} [{wl:+.1f},{wh:+.1f}]")
            D = np.zeros((B + 1, len(gens))); base = np.zeros((2, len(gens)))
            for b in range(B + 1):
                c = cal if b == 0 else rng.choice(cal, len(cal))
                for j, g in enumerate(gens):
                    f = fk[g] if b == 0 else rng.choice(fk[g], len(fk[g])); t = {}
                    for w in ("dv2", "dv4"):
                        t[w] = np.mean([(x[f] > ap.conformal_tau(x[c], .05)).mean() * 100 for x in S[w]])
                    D[b, j] = t["dv4"] - t["dv2"]
                    if b == 0:
                        base[:, j] = t["dv2"], t["dv4"]
            for j, g in enumerate(gens):
                lo, hi = np.percentile(D[1:, j], [2.5, 97.5])
                L.append(f"    {g:26s} {base[0, j]:5.1f} -> {base[1, j]:5.1f}   {D[0, j]:+6.1f} [{lo:+.1f},{hi:+.1f}]")
            print("\n".join(L[-len(gens) - 1:]), flush=True)
        L.append("")
    Path("runs/dv4/e8_compare.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    dump(sys.argv[2]) if sys.argv[1] == "dump" else compare()
