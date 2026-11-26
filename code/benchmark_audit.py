"""E2: metadata-shortcut audit of detection benchmarks (no pixels are read beyond the file header).

For each benchmark, every evaluation image is described only by its file metadata, and a gradient-boosted classifier
(5-fold stratified CV) tries to tell real from generated. AUROC 0.5 = the metadata carries no label information.
Feature groups:
  geometry     width, height, aspect ratio (survives any re-save; removed only by size matching)
  format       container format, JPEG quantisation (mean of table 0), chroma subsampling, progressive flag, colour mode
  provenance   EXIF present, number of EXIF tags, ICC profile present
  bytes/pixel  file size / pixel count (content-dependent compressibility; reported, not a pure shortcut)
  all          everything above
Each benchmark is audited as distributed, and also per generator (real images vs one generator's images): the
per-generator table shows where a pooled AUROC hides a perfect shortcut for one generator.

    anchor/bin/python -I benchmark_audit.py            -> runs/audit/{features_<bench>.npz, audit.txt, per_generator.txt}
    anchor/bin/python -I benchmark_audit.py --bench genimage chameleon
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "audit"
RAISE_TIFF = Path.home() / "anchor_data" / "raise" / "tiff"
FMT = {"JPEG": 0, "PNG": 1, "WEBP": 2, "TIFF": 3, "BMP": 4, "GIF": 5}
MODE = {"RGB": 0, "RGBA": 1, "L": 2, "P": 3, "CMYK": 4, "LA": 5, "I;16": 6}
GROUPS = {"geometry": [0, 1, 2], "format": [3, 4, 5, 6, 7], "provenance": [8, 9, 10], "bytes/pixel": [11],
          "all": list(range(12))}
COLS = ["width", "height", "aspect", "format", "jpeg_q0", "subsampling", "progressive", "mode",
        "has_exif", "n_exif", "has_icc", "bytes_per_px"]


def manifest(path, splits=("calib", "test")):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    return [(r["path"] if os.path.isabs(r["path"]) else str(ROOT / r["path"]), int(r["label"]), r["source"]) for r in rows]


def synthbuster_as_distributed():
    """Generated images as distributed plus each RAISE photograph once, as distributed (TIFF)."""
    rows = manifest(ROOT / "runs/sb/manifest.csv")
    fakes = [r for r in rows if r[1] == 1]
    ids = sorted({Path(p).stem for p, lab, _ in rows if lab == 0})
    reals = []
    for i in ids:
        t = next((RAISE_TIFF / f"{i}{e}" for e in (".TIF", ".tif", ".tiff") if (RAISE_TIFF / f"{i}{e}").exists()), None)
        if t is not None:
            reals.append((str(t), 0, "raise"))
    return reals + fakes


BENCHROOT = Path.home() / "anchor_data" / "bench"


def folder_rows(root, pattern_real="0_real", pattern_fake="1_fake"):
    """Rows from a <source>/{0_real,1_fake}/ tree; source = the folder above the class folder."""
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff") and p.parent.name in (pattern_real, pattern_fake):
            rows.append((str(p), int(p.parent.name == pattern_fake), p.parent.parent.name))
    return rows


def commfor_rows():
    """Community Forensics eval: sampled parquet shards (every 20th); each image's ORIGINAL file bytes are stored."""
    import pyarrow.parquet as pq
    rows = []
    for f in sorted((BENCHROOT / "commfor").glob("*.parquet")):
        try:
            t = pq.read_table(f, columns=["image_name", "label", "model_name", "real_source"])
        except Exception:                          # shard still downloading or truncated
            continue
        for k, (n, lab, mn, rs) in enumerate(zip(*[t.column(c).to_pylist() for c in ("image_name", "label", "model_name", "real_source")])):
            rows.append((f"{f}#{k}", int(lab), (mn or rs or "unknown") if lab == 1 else (rs or "real")))
    return rows


def source_rows(root, real="real"):
    """Rows from a <source>/ tree in which one source folder holds the real images."""
    rows = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    rows.append((str(p), int(d.name != real), d.name))
    return rows


def genimage_subset():
    """The E1 GenImage subset rows (runs/p4s/aeroblade/idx.npy) as distributed: the rows of the size-matched version."""
    rows = manifest(ROOT / "runs/p2/manifest.csv", splits=("train", "calib", "test"))
    import numpy as np
    return [rows[i] for i in np.load(ROOT / "runs/p4s/aeroblade/idx.npy")]


BENCH = {
    "robust_ldm_original": lambda: source_rows(BENCHROOT / "robust_ldm" / "images" / "original"),
    "robust_ldm_postprocessed": lambda: source_rows(BENCHROOT / "robust_ldm" / "images" / "post-processed"),
    "wildrf_test": lambda: folder_rows(BENCHROOT / "wildrf" / "WildRF" / "test"),
    "commfor_eval_sample": lambda: commfor_rows() if (Path.home() / "Desktop/FakeDetection/runs/bench_download.log").read_text().count("BENCH DOWNLOAD FINISHED") else [],
    "genimage": lambda: manifest(ROOT / "runs/p2/manifest.csv"),
    "chameleon": lambda: manifest(ROOT / "runs/cham/manifest.csv"),
    "chameleon_size_matched": lambda: manifest(ROOT / "runs/cham_m/manifest.csv"),
    "synthbuster_as_distributed": synthbuster_as_distributed,
    "synthbuster_downscaled": lambda: manifest(ROOT / "runs/sb/manifest.csv"),
    "synthbuster_cropped": lambda: manifest(ROOT / "runs/sb_crop/manifest.csv"),
    "genimage_e1_subset": lambda: genimage_subset(),
    "genimage_processing_control": lambda: manifest(ROOT / "runs/gi_c/manifest.csv"),
    "genimage_size_matched": lambda: manifest(ROOT / "runs/gi_m/manifest.csv"),
    "evalgen_downscaled": lambda: manifest(ROOT / "runs/eg/manifest.csv"),
    "evalgen_cropped": lambda: manifest(ROOT / "runs/eg_crop/manifest.csv"),
}


def _open(path):
    """A file path, or '<parquet>#<row>' for images stored as bytes (Community Forensics). Returns (file object, n_bytes)."""
    import io
    if "#" in path and path.split("#")[0].endswith(".parquet"):
        import pyarrow.parquet as pq
        f, k = path.split("#"); k = int(k)
        if _open.cache.get("f") != f:
            _open.cache = {"f": f, "t": pq.read_table(f, columns=["image_data"]).column("image_data")}
        b = _open.cache["t"][k].as_py(); return io.BytesIO(b), len(b)
    return path, Path(path).stat().st_size
_open.cache = {}


def describe(path):
    from PIL import Image
    src, nbytes = _open(path)
    with Image.open(src) as im:
        w, h = im.size; fmt = FMT.get(im.format, 9); mode = MODE.get(im.mode, 9)
        qt = getattr(im, "quantization", None); q0 = float(np.mean(qt[0])) if qt else -1.0
        sub = -1
        if im.format == "JPEG":
            from PIL import JpegImagePlugin
            sub = JpegImagePlugin.get_sampling(im)
        prog = int(bool(im.info.get("progressive") or im.info.get("progression")))
        try:
            ex = im.getexif(); n_ex = len(ex)
        except Exception:
            n_ex = 0
        icc = int(bool(im.info.get("icc_profile")))
    return [w, h, h / w, fmt, q0, sub, prog, mode, int(n_ex > 0), n_ex, icc, nbytes / (w * h)]


def features(name):
    f = OUT / f"features_{name}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False); return d["X"], d["y"], d["src"]
    rows = BENCH[name]()
    if not rows:                                  # data not downloaded yet
        return None
    X, y, src = [], [], []
    for k, (p, lab, s) in enumerate(rows):
        X.append(describe(p)); y.append(lab); src.append(s)
        if k % 5000 == 0:
            print(f"  {name}: {k}/{len(rows)}", flush=True)
    X, y, src = np.array(X, float), np.array(y), np.array(src)
    np.savez(f, X=X, y=y, src=src); return X, y, src


def auroc_cv(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan")
    if np.all(X == X[0]):
        return 0.5
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    import pandas as pd
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--bench", nargs="+", default=list(BENCH))
    a = ap_.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    pooled, per = {}, []
    for name in a.bench:
        F = features(name)
        if F is None:
            print(f"{name}: no data yet, skipped", flush=True); continue
        X, y, src = F
        pooled[name] = {g: auroc_cv(X[:, c], y) for g, c in GROUPS.items()}
        pooled[name]["n_real"], pooled[name]["n_generated"] = int((y == 0).sum()), int((y == 1).sum())
        for g in sorted(set(src[y == 1])):
            m = (y == 0) | (src == g)
            per.append(dict(benchmark=name, generator=g, **{k: auroc_cv(X[m][:, c], y[m]) for k, c in GROUPS.items()}))
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in pooled[name].items()}, flush=True)
    P = pd.DataFrame(pooled).T
    head = ["E2 metadata-shortcut audit: AUROC of a 5-fold CV gradient-boosted classifier that sees only file metadata",
            "(0.5 = no information). 'as distributed' = original files; '_downscaled' / '_cropped' / '_size_matched' = our matched versions.",
            "Evaluation rows only (calibration and test splits). Columns: " + ", ".join(f"{g} = {'/'.join(COLS[i] for i in c)}" for g, c in GROUPS.items() if g != "all"), ""]
    txt = "\n".join(head + [P.to_string(float_format=lambda v: f"{v:.3f}")])
    (OUT / "audit.txt").write_text(txt + "\n"); print(txt)
    Q = pd.DataFrame(per).set_index(["benchmark", "generator"])
    (OUT / "per_generator.txt").write_text("Real images vs one generator's images, same feature groups (AUROC)\n\n" +
                                           Q.to_string(float_format=lambda v: f"{v:.3f}") + "\n")


if __name__ == "__main__":
    main()
