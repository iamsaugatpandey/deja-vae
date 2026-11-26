"""E1 statistics: 95% bootstrap intervals for every detector under every benchmark version, and paired intervals for
(a) the change of each detector between versions of the same benchmark and (b) the difference between detectors.

Resampling unit: Synthbuster, the RAISE photograph (its matched reals and the generated images made from its description
move together, as in sb_cluster.py); GenImage and Chameleon, the image. Calibration reals and generated images are drawn
independently; the threshold is re-fit in every draw (split conformal, 5% FPR); TPR is the mean over generators.
The versions of one benchmark share their rows, so one draw is applied to all versions and all detectors (paired).
Multi-seed methods (Deja VAE-I) are seed-averaged inside each draw. B = 1,000, seed 0.
    anchor/bin/python e1_stats.py   -> runs/e1/stats.txt, runs/e1/stats_tpr.csv, runs/e1/stats_pairs.csv
"""
import csv, itertools, os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from e1_score import SETS, ROOT
from e1_report import FAMILIES, NAMES, ours

B, EPS = 1000, 0.05
CHANGES = {"GenImage": [("genimage_orig", "genimage_q95", "format cue removed (JPEG q95 re-save)")],
           "Chameleon": [("cham", "cham_c", "one resize + JPEG q75 (processing-only control)"),
                         ("cham_c", "cham_m", "width shortcut removed (size-matched vs control)")],
           "Synthbuster": [("sb_raw", "sb_q95", "q95 re-save"), ("sb_q95", "sb_crop_q95", "reals cropped instead of downscaled")],
           "EvalGEN": [("eg_q95", "eg_crop_q95", "reals cropped instead of downscaled")]}


def rows_and_scores(name):
    man, idxf, q = SETS[name]
    R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idxf)
    lab = np.array([int(R[i]["label"]) for i in I]); split = np.array([R[i]["split"] for i in I])
    src = np.array([R[i]["source"] for i in I]); pid = np.array([Path(R[i]["path"]).stem for i in I])
    S = {}
    for d in sorted(p.name for p in (ROOT / "runs/e1").iterdir() if p.is_dir()):
        f = ROOT / "runs/e1" / d / f"scores_{name}.npy"
        if f.exists():
            assert (np.load(ROOT / "runs/e1" / d / f"idx_{name}.npy") == I).all()
            S[d] = [np.load(f)]
    S.update(ours(name, I))
    return lab, split, src, pid, S


def draws(lab, split, src, pid, cluster, rng):
    cal = np.where((split == "calib") & (lab == 0))[0]; fk = np.where(lab == 1)[0]
    def groups(ix):
        if not cluster:
            return [np.array([i]) for i in ix]
        d = {}
        for i in ix:
            d.setdefault(pid[i], []).append(i)
        return [np.array(v) for v in d.values()]
    C, F = groups(cal), groups(fk)
    out = [(cal, fk)]                                                        # draw 0 = the point estimate
    for _ in range(B):
        out.append((np.concatenate([C[k] for k in rng.integers(0, len(C), len(C))]),
                    np.concatenate([F[k] for k in rng.integers(0, len(F), len(F))])))
    return out


def tpr(scores, src, gens, c, f):
    v = []
    for x in scores:
        t = ap.conformal_tau(x[c], EPS)
        v.append(np.mean([(x[f[src[f] == g]] > t).mean() * 100 for g in gens]))
    return float(np.mean(v))


def ci(v):
    return v[0], float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))


def main():
    L = [__doc__.strip().splitlines()[0], ""]; T, P = [], []
    for fam, names in FAMILIES:
        data = {n: rows_and_scores(n) for n in names}
        lab, split, src, pid, _ = data[names[0]]
        for n in names[1:]:
            assert (data[n][0] == lab).all() and (data[n][1] == split).all() and (data[n][2] == src).all(), f"{n} rows differ"
        gens = sorted(set(src[lab == 1])); D = draws(lab, split, src, pid, fam == "Synthbuster", np.random.default_rng(0))
        V = {(n, d): np.array([tpr(S, src, gens, c, f) for c, f in D]) for n in names for d, S in data[n][4].items()}
        L.append(f"=== {fam}  ({'clustered by RAISE photograph' if fam == 'Synthbuster' else 'image-level'} bootstrap, B={B})")
        for n in names:
            L.append(f"  [{n}]")
            for d in sorted({d for m, d in V if m == n}, key=lambda d: -V[(n, d)][0]):
                m, lo, hi = ci(V[(n, d)]); T.append([fam, n, NAMES.get(d, d), m, lo, hi])
                L.append(f"    {NAMES.get(d, d):26s} {m:5.1f} [{lo:5.1f}, {hi:5.1f}]")
        L.append("  Change between versions (later - earlier), paired:")
        for a, b, what in CHANGES[fam]:
            L.append(f"    {a} -> {b}: {what}")
            for d in sorted({d for m, d in V if m == a} & {d for m, d in V if m == b}):
                m, lo, hi = ci(V[(b, d)] - V[(a, d)]); P.append([fam, "change", f"{a}->{b}", NAMES.get(d, d), "", m, lo, hi])
                L.append(f"      {NAMES.get(d, d):26s} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        L.append("  Detector differences per version (row minus column), paired; order flips marked:")
        dets = sorted({d for _, d in V})
        for x, y in itertools.combinations(dets, 2):
            cells = [(n, ci(V[(n, x)] - V[(n, y)])) for n in names if (n, x) in V and (n, y) in V]
            if len(cells) < 2:
                continue
            signs = {np.sign(c[0]) for _, c in cells}; sig = {np.sign(c[0]) for _, c in cells if c[1] > 0 or c[2] < 0}
            flag = "FLIP (both directions significant)" if len(sig) == 2 else ("flip (point estimates)" if len(signs) == 2 else "")
            L.append(f"    {NAMES.get(x, x)} - {NAMES.get(y, y)}: " + "; ".join(f"{n} {c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}]" for n, c in cells) + (f"   {flag}" if flag else ""))
            for n, c in cells:
                P.append([fam, "pair", n, NAMES.get(x, x), NAMES.get(y, y), *c])
        L.append(""); print("\n".join(L[-60:]), flush=True)
    (ROOT / "runs/e1/stats.txt").write_text("\n".join(L) + "\n")
    with open(ROOT / "runs/e1/stats_tpr.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["benchmark", "version", "detector", "tpr5", "lo", "hi"]); w.writerows(T)
    with open(ROOT / "runs/e1/stats_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["benchmark", "kind", "version", "a", "b", "diff", "lo", "hi"]); w.writerows(P)


if __name__ == "__main__":
    main()
