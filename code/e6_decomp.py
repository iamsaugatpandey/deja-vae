"""E6 decomposition: how much of each detector's Chameleon TPR@5% came from the size/aspect shortcut?
Three versions of the SAME rows (reals identical; fakes differ):
  original  runs/cham_off   fakes as released
  control   runs/cham_c     same resize factor + JPEG q75 as matched, own aspect, full content
  matched   runs/cham_m     random real geometry: crop to its aspect, resize to 500 x H, JPEG q75
processing effect = original - control;  geometry (shortcut) effect = control - matched;  total = original - matched.
Paired bootstrap (B=1000): each draw resamples calibration reals and test fakes ONCE and applies the same
indices to all three versions and all seeds, so the intervals are for the paired, seed-averaged differences.

    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python e6_decomp.py   -> runs/cham_m/decomp.txt, decomp.csv, decomp_boot.npz
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv

VERS = {"original": ("cham", "runs/cham_off"), "control": ("cham_c", "runs/cham_c"), "matched": ("cham_m", "runs/cham_m")}
METHODS = ["simlbr_official", "lbr_exact", "sup", "recon", "dv", "dv-single", "dv-tf", "rigid-dinov3", "dv+fuse-simlbr_official"]
B = 1000


def load_all():
    rows, idx = dv.rows_of("cham"); out = {}
    for v, (sname, folder) in VERS.items():
        r, ix = dv.rows_of(sname); assert ix == idx
        dvr = dv.scores_for(sname, r, ix); S = {}
        for m in METHODS:
            if m in dvr:
                S[m] = [dvr[m][s][0] for s in (0, 1, 2)]
            elif "+fuse-" in m:
                continue
            else:
                S[m] = [np.load(f"{folder}/scores_{m}_seed{s}.npy") for s in (0, 1, 2)]
        d, l = S["dv"], S["simlbr_official"]                 # fusion exactly as in dejavae.cmd_report
        S["dv+fuse-simlbr_official"] = [dv.zfuse(d[s], l[s], r)(d[s], l[s]) for s in (0, 1, 2)]
        out[v] = S
    return rows, out


def main():
    import pandas as pd
    rows, S = load_all(); lab, split, _ = ap.split_idx(rows)
    cal = np.where((split == "calib") & (lab == 0))[0]; fk = np.where((split == "test") & (lab == 1))[0]
    rng = np.random.default_rng(0)
    draws = [(rng.choice(cal, len(cal)), rng.choice(fk, len(fk))) for _ in range(B)]

    def tpr(sc, c, f):
        return 100 * float((sc[f] > ap.conformal_tau(sc[c], 0.05)).mean())

    tab, boot = [], {}
    for m in METHODS:
        pt = {v: np.mean([tpr(s, cal, fk) for s in S[v][m]]) for v in VERS}
        bt = {v: np.array([np.mean([tpr(s, c, f) for s in S[v][m]]) for c, f in draws]) for v in VERS}
        eff = {"processing": bt["original"] - bt["control"], "geometry": bt["control"] - bt["matched"],
               "total": bt["original"] - bt["matched"]}
        row = dict(method=m, **{v: round(pt[v], 1) for v in VERS})
        for k, e in eff.items():
            p = {"processing": pt["original"] - pt["control"], "geometry": pt["control"] - pt["matched"],
                 "total": pt["original"] - pt["matched"]}[k]
            row[k] = f"{p:+.1f} [{np.percentile(e, 2.5):+.1f}, {np.percentile(e, 97.5):+.1f}]"
            boot[f"{m}|{k}"] = e
        for v in VERS:
            boot[f"{m}|{v}"] = bt[v]
        tab.append(row)
    df = pd.DataFrame(tab).set_index("method")
    O = Path("runs/cham_m")
    lines = ["E6 decomposition, Chameleon TPR@5% (threshold from calib reals; 3 seeds; paired bootstrap B=1000, 95% CI)",
             "processing = original - control (resize + JPEG q75, own aspect);  geometry = control - matched (shortcut);  total = original - matched",
             f"n_calib_real={len(cal)}  n_test_fake={len(fk)}", "", df.to_string()]
    txt = "\n".join(lines); print(txt)
    (O / "decomp.txt").write_text(txt + "\n"); df.to_csv(O / "decomp.csv"); np.savez(O / "decomp_boot.npz", **boot)


if __name__ == "__main__":
    main()
