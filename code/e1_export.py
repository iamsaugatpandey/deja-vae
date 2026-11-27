"""Export per-image scores of the re-evaluation (E1: released detectors and our methods on every benchmark version) and of
the false-alarm test on phone photographs (E5, VISION), plus the matching lists of size-matched GenImage and EvalGEN.
    anchor/bin/python e1_export.py OUT_DIR   -> OUT_DIR/scores/e1_<version>.csv.gz, OUT_DIR/scores/e5_vision.csv.gz,
                                                OUT_DIR/matching/genimage_size_matched.csv, OUT_DIR/matching/evalgen_reals.csv
Image ids are paths relative to each dataset's root (no local directories). Scores: higher = more likely generated;
released detectors as produced by their authors' code (logits), ours as in the paper (Deja VAE-I: three seeds).
"""
import csv, gzip, os, sys
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e1_score import SETS, ROOT
from e1_report import ours

OUT = Path(sys.argv[1]); (OUT / "scores").mkdir(parents=True, exist_ok=True); (OUT / "matching").mkdir(parents=True, exist_ok=True)
MARKERS = ("genimage/", "chameleon_data/", "synthbuster_real_matched/", "synthbuster_real_cropped/", "synthbuster/synthbuster/",
           "cham_matched/", "cham_ctrl/", "evalgen_real_matched/", "evalgen_real_cropped/", "EvalGEN/", "gi_matched/", "gi_ctrl/",
           "vision/")
RELEASED = ["bfree", "dda", "npr", "alignedforensics", "spai"]
PUBLIC = {"genimage_orig": "genimage_as_distributed", "genimage_q95": "genimage_q95", "gi_c": "genimage_processing_control",
          "gi_m": "genimage_size_matched", "cham": "chameleon", "cham_c": "chameleon_processing_control",
          "cham_m": "chameleon_size_matched", "sb_raw": "synthbuster_downscaled_no_resave", "sb_q95": "synthbuster_downscaled",
          "sb_crop_q95": "synthbuster_cropped", "eg_q95": "evalgen_downscaled", "eg_crop_q95": "evalgen_cropped", "e5": "e5_vision"}


def rel(p):
    for m in MARKERS:
        if m in p:
            return p[p.index(m):]
    raise ValueError(f"no dataset marker in {p}")


def main():
    for name, (man, idxf, q) in SETS.items():
        R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idxf)
        cols = {}
        for d in RELEASED:
            f = ROOT / "runs/e1" / d / f"scores_{name}.npy"
            if f.exists():
                cols[d] = np.load(f)
        O = ours(name, I)
        if name == "e5":                                   # E5 is scored outside the E1 benchmark map (e5_report.py)
            import dejavae as dv
            O = {"simlbr": [np.load(ROOT / "runs/e5/scores_simlbr_official_seed0.npy")[I]]}
            rows, idx = dv.rows_of("e5"); D = dv.scores_for("e5", rows, idx)
            O["dvI"] = [D["dv-single"][s][0][I] for s in (0, 1, 2)]
        for k, v in O.items():
            for s, x in enumerate(v):
                cols[f"{'SimLBR_released' if k == 'simlbr' else 'DejaVAE_I'}@seed{s}"] = x
        extra = ["device", "scene"] if "device" in R[0] else []
        fn = OUT / "scores" / (f"{PUBLIC[name]}.csv.gz" if name == "e5" else f"e1_{PUBLIC[name]}.csv.gz")
        with gzip.open(fn, "wt", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "split", "label", "source", *extra, *cols])
            for j, i in enumerate(I):
                r = R[i]; w.writerow([rel(r["path"]), r["split"], r["label"], r["source"], *[r[e] for e in extra],
                                      *[f"{cols[c][j]:.6g}" for c in cols]])
        print(f"{fn.name}: {len(I)} rows, {len(cols)} score columns", flush=True)
    log = np.load(ROOT / "runs/gi_m/match_log.npy")
    fk = [r for r in csv.DictReader(open(ROOT / "runs/gi_m/manifest.csv")) if r["label"] == "1"]
    with open(OUT / "matching/genimage_size_matched.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["matched_image", "source", "orig_w", "orig_h", "crop_w", "crop_h", "target_w", "target_h"])
        for r, l in zip(fk, log):
            w.writerow([rel(r["path"]), r["source"], *l])
    reals = [r for r in csv.DictReader(open(ROOT / "runs/eg/manifest.csv")) if r["label"] == "0"]
    with open(OUT / "matching/evalgen_reals.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["raise_id", "split"])
        for r in reals:
            w.writerow([Path(r["path"]).stem, r["split"]])
    print("matching lists written")


if __name__ == "__main__":
    main()
