"""E1 report: released detectors and the paper's own methods on the same subset rows, under original and matched versions.
Threshold: split conformal at 5% FPR on the subset's calibration reals (as in the paper). Metrics: mean and worst TPR@5%
over generators, mean AUROC (test reals vs each generator), FPR on test reals. Ranks within each benchmark version.
    anchor/bin/python e1_report.py   -> runs/e1/report.txt, runs/e1/metrics.csv
"""
import csv, os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from e1_score import SETS, ROOT
from dv_bank_full import auroc

FAMILIES = [("GenImage", ["genimage_orig", "genimage_q95", "gi_c", "gi_m"]), ("Chameleon", ["cham", "cham_c", "cham_m"]),
            ("Synthbuster", ["sb_raw", "sb_q95", "sb_crop_q95"]), ("EvalGEN", ["eg_q95", "eg_crop_q95"])]
OURS_SET = {"genimage_q95": "genimage", "cham": "cham", "cham_c": "cham_c", "cham_m": "cham_m", "sb_q95": "sb", "sb_crop_q95": "sb_crop",
            "eg_q95": "eg", "eg_crop_q95": "eg_crop", "gi_c": "gi_c", "gi_m": "gi_m"}
NAMES = {"bfree": "B-Free", "dda": "DDA", "npr": "NPR", "alignedforensics": "AlignedForensics", "spai": "SPAI", "aide": "AIDE",
         "simlbr": "SimLBR (released)", "dvI": "Deja VAE-I", "dvI_final": "Deja VAE-I (final layer)"}


def ours(name, I):
    """Scores of the paper's methods on the subset rows I of benchmark `name` (None if not scored under this version)."""
    if name not in OURS_SET:
        return {}
    import dejavae as dv
    st = OURS_SET[name]; rows, idx = dv.rows_of(st); ref = Path(dv.SETS[st][1]); out = {}
    if (ref / "scores_simlbr_official_seed0.npy").exists():
        out["simlbr"] = [np.load(ref / "scores_simlbr_official_seed0.npy")[I]]
    S = dv.scores_for(st, rows, idx)
    if "dv-single" in S:
        out["dvI"] = [S["dv-single"][s][0][I] for s in (0, 1, 2)]
    return out


def metrics(scores, lab, split, src):
    cal = (split == "calib") & (lab == 0); te = (split == "test") & (lab == 0)
    gens = sorted(set(src[lab == 1])); res = []
    for x in scores:
        tau = ap.conformal_tau(x[cal], 0.05)
        tpr = [100 * (x[(lab == 1) & (src == g)] > tau).mean() for g in gens]
        au = [auroc(x[(lab == 1) & (src == g)], x[te]) for g in gens]
        res.append([np.mean(tpr), np.min(tpr), np.mean(au), 100 * (x[te] > tau).mean()])
    return np.mean(res, 0)


def main():
    dets = [d.name for d in sorted((ROOT / "runs" / "e1").iterdir()) if d.is_dir()]
    L = [__doc__.strip().splitlines()[0], ""]; table = []
    for fam, names in FAMILIES:
        L.append(f"=== {fam}")
        for name in names:
            man, idxf, q = SETS[name]
            R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idxf)
            lab = np.array([int(R[i]["label"]) for i in I]); split = np.array([R[i]["split"] for i in I]); src = np.array([R[i]["source"] for i in I])
            S = {d: [np.load(ROOT / "runs/e1" / d / f"scores_{name}.npy")] for d in dets if (ROOT / "runs/e1" / d / f"scores_{name}.npy").exists()}
            S.update(ours(name, I))
            if not S:
                continue
            M = {k: metrics(v, lab, split, src) for k, v in S.items()}
            order = sorted(M, key=lambda k: -M[k][0])
            L.append(f"  [{name}]  (re-save: {q or 'none'})")
            for r, k in enumerate(order, 1):
                m = M[k]; table.append(dict(benchmark=name, method=NAMES.get(k, k), rank=r, tpr5=m[0], worst=m[1], auroc=m[2], fpr_test_reals=m[3]))
                L.append(f"    {r}. {NAMES.get(k, k):26s} TPR@5 {m[0]:5.1f}  worst {m[1]:5.1f}  AUROC {m[2]:.3f}  FPR(test reals) {m[3]:4.1f}")
        L.append("")
    txt = "\n".join(L); print(txt); (ROOT / "runs/e1/report.txt").write_text(txt + "\n")
    with open(ROOT / "runs/e1/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


if __name__ == "__main__":
    main()
