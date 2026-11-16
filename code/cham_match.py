"""E6: size-matched Chameleon. Every Chameleon real is a 500-px-wide, baseline JPEG q75 with 4:2:0 chroma and
no EXIF; the fakes have other widths and aspect ratios (median H/W 0.72 real vs 1.5 fake), but are also mostly
JPEG q75. Width alone separates the classes (TPR 99.98 at FPR 0).

Matched protocol: each fake gets the geometry of a randomly drawn real (seed 17): centre-crop to that real's
aspect ratio, resize to exactly 500 x H_real (PIL bicubic, antialiased), save as baseline JPEG q75, 4:2:0, no
metadata. Reals are untouched. The size distributions of the two classes are then identical by construction;
`audit` checks what metadata can still tell them apart.

Rows keep the order and splits of runs/cham/manifest.csv, so real-image features and scores are copied from
the original folders (`seed`) and only the 11,170 fakes are recomputed.

    anchor/bin/python cham_match.py make     # images -> ~/anchor_data/cham_matched, manifest -> runs/cham_m
    anchor/bin/python cham_match.py seed     # copy real-row features/pairs into runs/cham_m and runs/dv2
    anchor/bin/python cham_match.py audit    # metadata-only classifiers, original vs matched -> runs/cham_m/audit.txt
"""
import sys, json, csv, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap

SRC = Path("runs/cham/manifest.csv"); OUT = Path("runs/cham_m"); IMG = Path.home() / "anchor_data" / "cham_matched"
Q = 75


def make():
    from PIL import Image
    from tqdm import tqdm
    OUT.mkdir(parents=True, exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC)))
    reals = [r for r in rows if r["label"] == "0"]
    rsz = []
    for r in reals:
        with Image.open(r["path"]) as im:
            rsz.append(im.size)
    rsz = np.array(rsz); assert (rsz[:, 0] == 500).all()
    rng = np.random.default_rng(17)
    fk = [i for i, r in enumerate(rows) if r["label"] == "1"]
    tgt = rsz[rng.integers(0, len(rsz), len(fk))]
    log = []
    for j, (i, (tw, th)) in enumerate(tqdm(list(zip(fk, tgt)), desc="match")):
        dst = IMG / (Path(rows[i]["path"]).stem + f"_{i}.jpg")
        with Image.open(rows[i]["path"]) as im0:
            w, h = im0.size; im = im0.convert("RGB")
        a = th / tw
        if h / w > a:                                   # too tall: crop height
            ch = max(1, round(w * a)); t = (h - ch) // 2; im = im.crop((0, t, w, t + ch))
        else:                                           # too wide: crop width
            cw = max(1, round(h / a)); l = (w - cw) // 2; im = im.crop((l, 0, l + cw, h))
        log.append([w, h, im.size[0], im.size[1], int(tw), int(th)])
        if not dst.exists():
            im.resize((int(tw), int(th)), Image.BICUBIC).save(dst, "JPEG", quality=Q, subsampling=2, optimize=False)
        rows[i] = dict(rows[i], path=str(dst))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    np.save(OUT / "match_log.npy", np.array(log))
    up = np.array(log); scale = up[:, 4] / up[:, 2]
    print(f"wrote {len(fk)} matched fakes to {IMG}; resize factor median {np.median(scale):.2f}, "
          f"upscaled {100 * (scale > 1).mean():.1f}%, kept area median {np.median(up[:, 2] * up[:, 3] / (up[:, 0] * up[:, 1])):.2f}")


def make_ctrl():
    """Processing-only control (runs/cham_c): each fake gets the SAME resize factor and JPEG q75 re-save as
    in the matched set, but keeps its own aspect ratio and full content (no crop). matched - control then
    isolates the geometry change (shortcut removal); original - control isolates the processing."""
    from PIL import Image
    from tqdm import tqdm
    C = Path("runs/cham_c"); D = Path.home() / "anchor_data" / "cham_ctrl"
    C.mkdir(parents=True, exist_ok=True); D.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC))); log = np.load(OUT / "match_log.npy")
    fk = [i for i, r in enumerate(rows) if r["label"] == "1"]
    for (i, (w, h, cw, ch, tw, th)) in tqdm(list(zip(fk, log)), desc="ctrl"):
        s = tw / cw; dst = D / (Path(rows[i]["path"]).stem + f"_{i}.jpg")
        if not dst.exists():
            with Image.open(rows[i]["path"]) as im0:
                im = im0.convert("RGB")
            im.resize((max(8, round(w * s)), max(8, round(h * s))), Image.BICUBIC).save(dst, "JPEG", quality=Q, subsampling=2, optimize=False)
        rows[i] = dict(rows[i], path=str(dst))
    with open(C / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"wrote {len(fk)} control fakes to {D}")


def seed(out=None, setname="cham_m"):
    """Real rows are identical: copy their features/pairs; fakes are recomputed by the normal pipelines."""
    out = Path(out or OUT)
    rows = ap.read_manifest(out / "manifest.csv"); N = len(rows)
    real = [i for i, r in enumerate(rows) if r["label"] == 0]
    old = ap.read_manifest(SRC)
    assert [r["path"] for r, o in zip(rows, old) if r["label"] == 0] == [o["path"] for o in old if o["label"] == 0]
    C = Path("runs/cham_off")
    if not (out / "cls.npy").exists():
        for f in ("cls.npy", "cls_degraded.npy", "cls_recon.npy", "features_meta.json"):
            shutil.copy(C / f, out / f)
        json.dump(real, open(out / "features_done.json", "w"))
        print("seeded features (anchor_pilot) for", len(real), "real rows")
    D = Path("runs/dv2")
    if not (D / f"{setname}_done.json").exists():
        for k in ("img", "deg"):
            for s in "ab":
                shutil.copy(D / f"cham_{k}_{s}.npy", D / f"{setname}_{k}_{s}.npy")
        json.dump(real, open(D / f"{setname}_done.json", "w"))
        print("seeded Déjà VAE pairs for", len(real), "real rows")


def meta_table(manifest):
    from PIL import Image
    rows = ap.read_manifest(manifest); X = []
    for r in rows:
        p = Path(r["path"])
        with Image.open(p) as im:
            w, h = im.size; fmt = {"JPEG": 0, "PNG": 1, "WEBP": 2}.get(im.format, 3)
            qt = getattr(im, "quantization", None); q0 = float(np.mean(qt[0])) if qt else -1.0
        X.append([w, h, h / w, fmt, q0, p.stat().st_size / (w * h)])
    return np.array(X), np.array([r["label"] for r in rows])


def audit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    lines = ["Metadata-only classifiers (5-fold CV gradient boosting, AUROC; 0.5 = no information)", ""]
    feats = {"width": [0], "width+height": [0, 1], "aspect": [2], "format+quant-table": [3, 4],
             "bytes/pixel": [5], "all six": [0, 1, 2, 3, 4, 5]}
    res = {}
    for name, man in (("original", SRC), ("matched", OUT / "manifest.csv")):
        X, y = meta_table(man); res[name] = {}
        for fn, cols in feats.items():
            p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X[:, cols], y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            res[name][fn] = roc_auc_score(y, p)
        rule = (X[:, 0] != 500)
        res[name]["rule width!=500"] = roc_auc_score(y, rule)
    import pandas as pd
    lines.append(pd.DataFrame(res).round(3).to_string())
    lines += ["", "bytes/pixel is content-dependent (compressibility of the image) and is reported, not removed."]
    txt = "\n".join(lines); print(txt); (OUT / "audit.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"make": make, "seed": seed, "audit": audit, "make_ctrl": make_ctrl,
     "seed_ctrl": lambda: seed("runs/cham_c", "cham_c")}[sys.argv[1]]()
