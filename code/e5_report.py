"""E5 report: false-alarm rate on everyday real photographs (VISION subset), per sharing version and detector.

Thresholds (all fixed before looking at VISION):
  own      the detector's own decision rule (logit > 0, i.e. probability > 0.5), as a user running the released code gets.
           Released detectors and the released SimLBR head; Deja VAE-I has no default rule.
  camera   5% FPR on camera photographs: split conformal on the calibration reals of Synthbuster's E1 subset (RAISE,
           downscaled), in the same pipeline the detector applies to VISION (released: files as stored, sb_raw;
           SimLBR / Deja VAE-I: their q95 pipeline, sb_q95).
  genimage 5% FPR on GenImage's calibration reals (ImageNet photographs; released: as distributed; ours: q95).
Intervals: 95% bootstrap over devices (all of a device's photographs move together; B = 1,000, seed 0). Version
changes (native -> Facebook / WhatsApp) are paired by scene.
    anchor/bin/python e5_report.py   -> runs/e5/report.txt, runs/e5/fpr.csv
"""
import csv, os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from e1_score import SETS, ROOT
from e1_report import NAMES, ours

VERS = [("nat", "native (as taken)"), ("natFBH", "Facebook, high quality"), ("natFBL", "Facebook, low quality"),
        ("natWA", "WhatsApp"), ("flat", "low texture (walls, skies), native")]
RELEASED = ["bfree", "dda", "npr", "alignedforensics", "spai"]


def calib(name, det):
    """Calibration-real scores of detector `det` on benchmark version `name` (E1 subset rows)."""
    man, idxf, _ = SETS[name]
    R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idxf)
    cal = np.array([(R[i]["split"] == "calib") and R[i]["label"] == "0" for i in I])
    S = [np.load(ROOT / "runs/e1" / det / f"scores_{name}.npy")] if det in RELEASED else ours(name, I)[det]
    return [s[cal] for s in S]


def main():
    import dejavae as dv
    R = list(csv.DictReader(open(ROOT / "runs/e5/manifest.csv")))
    src = np.array([r["source"] for r in R]); dev = np.array([r["device"] for r in R]); scn = np.array([r["scene"] for r in R])
    S = {d: [np.load(ROOT / "runs/e1" / d / "scores_e5.npy")] for d in RELEASED if (ROOT / "runs/e1" / d / "scores_e5.npy").exists()}
    if (ROOT / "runs/e5/scores_simlbr_official_seed0.npy").exists():
        S["simlbr"] = [np.load(ROOT / "runs/e5/scores_simlbr_official_seed0.npy")]
    rows, idx = dv.rows_of("e5"); D = dv.scores_for("e5", rows, idx)
    if "dv-single" in D:
        S["dvI"] = [D["dv-single"][s][0] for s in (0, 1, 2)]
    TH = {}
    for d in S:
        rel = d in RELEASED
        TH[(d, "own")] = [0.0] * len(S[d]) if d != "dvI" else None
        TH[(d, "camera")] = [ap.conformal_tau(c, 0.05) for c in calib("sb_raw" if rel else "sb_q95", d)]
        TH[(d, "genimage")] = [ap.conformal_tau(c, 0.05) for c in calib("genimage_orig" if rel else "genimage_q95", d)]
    devs = sorted(set(dev)); by_dev = {v: np.where(dev == v)[0] for v in devs}
    rng = np.random.default_rng(0)
    draws = [np.arange(len(R))] + [np.concatenate([by_dev[devs[k]] for k in rng.integers(0, len(devs), len(devs))]) for _ in range(1000)]

    def fpr(d, th, sel):
        return float(np.mean([(x[sel] > t).mean() * 100 for x, t in zip(S[d], th)]))

    L = [__doc__.strip().splitlines()[0], f"{len(R)} photographs from {len(devs)} devices", ""]; out = []
    for tname in ("own", "camera", "genimage"):
        L.append(f"=== threshold: {tname}")
        L.append(f"  {'detector':22s}" + "".join(f"{v:>22s}" for v, _ in VERS))
        for d in S:
            th = TH[(d, tname)]
            if th is None:
                continue
            cells = []
            for v, _ in VERS:
                vals = np.array([fpr(d, th, dr[src[dr] == v]) for dr in draws])
                lo, hi = np.percentile(vals[1:], [2.5, 97.5]); cells.append(f"{vals[0]:6.1f} [{lo:4.1f},{hi:5.1f}]")
                out.append([tname, NAMES.get(d, d), v, vals[0], lo, hi])
            L.append(f"  {NAMES.get(d, d):22s}" + "".join(f"{c:>22s}" for c in cells))
        L.append("  change from native, paired by scene (points):")
        for d in S:
            th = TH[(d, tname)]
            if th is None:
                continue
            cells = []
            for v in ("natFBH", "natFBL", "natWA"):
                diffs = []
                for dr in draws:
                    a = dr[src[dr] == "nat"]; b = dr[src[dr] == v]
                    diffs.append(fpr(d, th, b) - fpr(d, th, a))      # same scenes on both sides (all four versions kept)
                diffs = np.array(diffs); lo, hi = np.percentile(diffs[1:], [2.5, 97.5])
                cells.append(f"{v} {diffs[0]:+.1f} [{lo:+.1f},{hi:+.1f}]")
            L.append(f"    {NAMES.get(d, d):22s} " + ";  ".join(cells))
        L.append("")
    txt = "\n".join(L); print(txt); (ROOT / "runs/e5/report.txt").write_text(txt + "\n")
    with open(ROOT / "runs/e5/fpr.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["threshold", "detector", "version", "fpr", "lo", "hi"]); w.writerows(out)


if __name__ == "__main__":
    main()
