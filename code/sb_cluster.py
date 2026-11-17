"""Synthbuster paired bootstrap clustered by RAISE photograph (review: statistical unit).
Each RAISE photograph contributes one matched real per generator (nine rows; for Deja VAE on crop-matched reals the nine
views are pixel-identical) and up to nine generated images made from its description. Resampling rows treats these as
independent and narrows the intervals, so here each draw resamples photographs: calibration photographs from the
calibration half (all their rows enter the threshold) and, independently, photographs for the generated images (all
their generated images enter each generator's TPR). Otherwise as anchor_pilot.paired_bootstrap (seed-averaged; B=1000).
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python sb_cluster.py          -> runs/sb/cluster.txt (pooled comparisons)
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python sb_cluster.py pergen   -> runs/sb/cluster_pergen.txt (per generator)
"""
import os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent / "figures"))
import numpy as np
import anchor_pilot as ap
import fig_fusion as ff

PAIRS = [("dv-single", "simlbr_official"), ("dv-single", "sup"), ("dv-single", "recon"), ("dv", "dv-single"),
         ("dv", "simlbr_official"), ("dv+fuse-simlbr_official", "simlbr_official"),
         ("dv-single+fuse-simlbr_official", "simlbr_official"), ("dv-single+fuse-simlbr_official", "dv+fuse-simlbr_official"),
         ("dv+fuse-simlbr_official", "sup+fuse-simlbr_official"), ("dv+fuse-simlbr_official", "recon+fuse-simlbr_official"),
         ("dv+fuse-simlbr_official", "rigid-dinov2+fuse-simlbr_official"), ("sup+fuse-simlbr_official", "simlbr_official")]


def cluster_boot(sa, sb, rows, eps=0.05, B=1000, seed=0):
    lab, split, src = ap.split_idx(rows); pid = np.array([Path(r["path"]).stem for r in rows])
    gens = sorted(set(src[(split == "test") & (lab == 1)]))
    cal = np.where((split == "calib") & (lab == 0))[0]; fk = np.where((split == "test") & (lab == 1))[0]
    def by_id(ix):
        d = {}
        for i in ix:
            d.setdefault(pid[i], []).append(i)
        return list(d.values())
    C, Fk = by_id(cal), by_id(fk); rng = np.random.default_rng(seed); dm, dw = [], []
    for _ in range(B):
        c = np.concatenate([C[k] for k in rng.integers(0, len(C), len(C))])
        f = np.concatenate([Fk[k] for k in rng.integers(0, len(Fk), len(Fk))])
        fs = [f[src[f] == g] for g in gens]; m_d, w_d = [], []
        for xa, xb in zip(sa, sb):
            ta, tb = ap.conformal_tau(xa[c], eps), ap.conformal_tau(xb[c], eps)
            ra = np.array([(xa[x] > ta).mean() for x in fs]) * 100; rb = np.array([(xb[x] > tb).mean() for x in fs]) * 100
            m_d.append(ra.mean() - rb.mean()); w_d.append(ra.min() - rb.min())
        dm.append(np.mean(m_d)); dw.append(np.mean(w_d))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(mean=q(dm), worst=q(dw))


def pergen(sa, sb, rows, eps=0.05, B=1000, seed=1):
    lab, split, src = ap.split_idx(rows); pid = np.array([Path(r["path"]).stem for r in rows])
    gens = sorted(set(src[(split == "test") & (lab == 1)])); cal = np.where((split == "calib") & (lab == 0))[0]
    C = {}
    for i in cal:
        C.setdefault(pid[i], []).append(i)
    C = list(C.values()); rng = np.random.default_rng(seed); out = {}
    for g in gens:
        fk = np.where((split == "test") & (lab == 1) & (src == g))[0]; d = np.zeros(B + 1)
        for k in range(B + 1):
            c = cal if k == 0 else np.concatenate([C[j] for j in rng.integers(0, len(C), len(C))])
            f = fk if k == 0 else rng.choice(fk, len(fk))      # one generated image per photograph within a generator
            d[k] = np.mean([((xa[f] > ap.conformal_tau(xa[c], eps)).mean() - (xb[f] > ap.conformal_tau(xb[c], eps)).mean()) * 100
                            for xa, xb in zip(sa, sb)])
        out[g] = (d[0], *np.percentile(d[1:], [2.5, 97.5]))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pergen":
        L = ["Per-generator paired difference in TPR@5% (pooled threshold), calibration resampled by RAISE photograph, B=1000.", ""]
        for SET in ("sb", "sb_crop"):
            rows, runs = ff.runs_for(SET); S = lambda m: [runs[m][s][0] for s in (0, 1, 2)]
            for a, b in [("dv-single", "simlbr_official"), ("dv+fuse-simlbr_official", "simlbr_official"), ("dv", "dv-single"),
                         ("dv-single-nocf", "dv-single")]:
                L.append(f"[{SET}] {a} - {b}")
                for g, (m, lo, hi) in pergen(S(a), S(b), rows).items():
                    L.append(f"    {g:24s} {m:+6.1f} [{lo:+.1f},{hi:+.1f}]")
                print("\n".join(L[-10:]), flush=True)
            if "dv-single-nocf" in runs:
                r = cluster_boot(S("dv-single-nocf"), S("dv-single"), rows)
                L.append(f"[{SET}] pooled dv-single-nocf - dv-single: mean {r['mean'][0]:+.1f} [{r['mean'][1]:+.1f},{r['mean'][2]:+.1f}]")
            L.append("")
        Path("runs/sb/cluster_pergen.txt").write_text("\n".join(L) + "\n"); return

    L = ["Synthbuster paired bootstrap: rows (as in the report) vs clustered by RAISE photograph. TPR@5%, seed-averaged, B=1000.", ""]
    for SET in ("sb", "sb_crop"):
        rows, runs = ff.runs_for(SET); S = lambda m: [runs[m][s][0] for s in (0, 1, 2)]
        L.append(f"[{SET}]")
        for a, b in PAIRS:
            if a not in runs or b not in runs:
                continue
            r0 = ap.paired_bootstrap(S(a), S(b), rows, eps=0.05, B=1000); r1 = cluster_boot(S(a), S(b), rows)
            f = lambda r: f"mean {r['mean'][0]:+5.1f} [{r['mean'][1]:+.1f},{r['mean'][2]:+.1f}] worst {r['worst'][0]:+5.1f} [{r['worst'][1]:+.1f},{r['worst'][2]:+.1f}]"
            L.append(f"  {a:31s} - {b:33s} rows: {f(r0)}   photos: {f(r1)}"); print(L[-1], flush=True)
        L.append("")
    Path("runs/sb/cluster.txt").write_text("\n".join(L) + "\n")



if __name__ == "__main__":
    main()
