"""Reproduction checks: each released detector's own-rule result (logit > 0) on our evaluation subsets against published
numbers (DDA's authors, NeurIPS 2025, Tables 2/4/7/10, balanced accuracy; SPAI's authors, CVPR 2025, Table 1, AUROC).
Inputs: runs/e1/<detector>/scores_<set>.npy (e1_score.py; eg_raw_fake = EvalGEN generated images as distributed) and
runs/e1/npr/scores_<set>_crop224.npy (npr_variant_check.py).
    anchor/bin/python -I repro_check.py  -> runs/e1/repro_check.txt, OUT/scores/reproduction_checks.csv.gz
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import e1_score as E

OUT = ROOT.parent / "data"
PUB = {  # (benchmark set, detector): published value
    ("sb_raw", "dda"): 90.1, ("sb_raw", "alignedforensics"): 77.4, ("sb_raw", "npr"): 50.0,
    ("genimage_q95", "dda"): 91.7, ("genimage_q95", "alignedforensics"): 79.0, ("genimage_q95", "npr"): 51.5,
    ("cham", "dda"): 82.4, ("cham", "alignedforensics"): 71.0, ("cham", "npr"): 59.9,
    ("eg_raw_fake", "dda"): 97.2, ("eg_raw_fake", "alignedforensics"): 68.0, ("eg_raw_fake", "npr"): 2.9,
    ("eg_raw_fake", "npr_crop224"): 2.9, ("sb_raw", "npr_crop224"): 50.0,
}
SPAI_PUB = {"glide": 90.2, "stable-diffusion-1-3": 99.6, "stable-diffusion-1-4": 99.6, "dalle2": 91.1,
            "stable-diffusion-2": 96.5, "stable-diffusion-xl": 97.4, "midjourney-v5": 94.5, "dalle3": 90.2, "firefly": 96.0}


def meta(name):
    man, idx, _ = E.SETS[name]; R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idx)
    return I, [R[i]["path"] for i in I], np.array([int(R[i]["label"]) for i in I]), np.array([R[i]["source"] for i in I])


def scores(name, det):
    if det.endswith("_crop224"):
        return np.load(ROOT / f"runs/e1/{det[:-8]}/scores_{name}_crop224.npy")
    return np.load(ROOT / f"runs/e1/{det}/scores_{name}.npy")


def balanced(s, y, src):
    """Mean over generators of (generated accuracy + real accuracy) / 2; real accuracy over that generator's reals if
    the benchmark pairs them, else over all reals; generated accuracy alone if there are no reals."""
    out = []
    for g in sorted(set(src[y == 1])):
        fa = (s[(src == g) & (y == 1)] > 0).mean()
        rm = (y == 0) & (src == g) if ((y == 0) & (src == g)).any() else (y == 0)
        out.append((fa + (s[rm] <= 0).mean()) / 2 if rm.any() else fa)
    return 100 * float(np.mean(out))


def main():
    lines = ["Reproduction checks: own rule (logit > 0) on our evaluation subsets vs published numbers", "",
             f"{'set':14s} {'detector':20s} {'published':>9s} {'ours':>6s} {'diff':>6s}"]
    rows = []
    for (name, det), pub in PUB.items():
        I, P, y, src = meta(name); s = scores(name, det); ours = balanced(s, y, src)
        lines.append(f"{name:14s} {det:20s} {pub:9.1f} {ours:6.1f} {ours - pub:+6.1f}")
        rows += [dict(image=p, set=name, detector=det, label=int(l), source=g, score=float(v)) for p, l, g, v in zip(P, y, src, s)]
    I, P, y, src = meta("sb_raw"); s = scores("sb_raw", "spai")
    au = {g: 100 * roc_auc_score(y[src == g], s[src == g]) for g in SPAI_PUB}
    lines += ["", "SPAI AUROC x100 per Synthbuster generator (ours: RAISE reals only; published: mean over five real sources)"]
    lines += [f"  {g:22s} published {SPAI_PUB[g]:5.1f}  ours {au[g]:5.1f}" for g in SPAI_PUB]
    lines.append(f"  {'mean':22s} published {np.mean(list(SPAI_PUB.values())):5.1f}  ours {np.mean(list(au.values())):5.1f}")
    txt = "\n".join(lines) + "\n"; (ROOT / "runs/e1/repro_check.txt").write_text(txt); print(txt)
    (OUT / "results").mkdir(parents=True, exist_ok=True); (OUT / "results/reproduction_checks.txt").write_text(txt)
    import pandas as pd
    markers = ("genimage/", "chameleon_data/", "synthbuster/", "synthbuster_real_matched/", "EvalGEN/", "evalgen_real_matched/")
    rel = lambda p: next(p[p.index(m):] for m in markers if m in p)
    d = pd.DataFrame(rows); d["image"] = d["image"].map(rel)
    d[d.detector.isin(["npr_crop224"]) | (d.set == "eg_raw_fake")].to_csv(OUT / "scores/reproduction_checks.csv.gz", index=False)


if __name__ == "__main__":
    main()
