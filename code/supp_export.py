"""Export the per-image scores and matching lists that go into the supplementary material (no images).
    DV_FEAT=mid anchor/bin/python supp_export.py OUT_DIR
Writes OUT_DIR/scores/<benchmark>.csv.gz, OUT_DIR/scores/aeroblade_<benchmark>.csv.gz and OUT_DIR/matching/*.csv.
Image ids are paths relative to each dataset's root, so they contain no local directories.
"""
import os, sys, csv, gzip
from pathlib import Path
os.environ.setdefault("DV_FEAT", "mid")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap

OUT = Path(sys.argv[1]); (OUT / "scores").mkdir(parents=True, exist_ok=True); (OUT / "matching").mkdir(parents=True, exist_ok=True)
MARKERS = ("genimage/", "chameleon_data/", "synthbuster_real_matched/", "synthbuster_real_cropped/", "synthbuster/synthbuster/", "cham_matched/", "cham_ctrl/")
BASE = {"simlbr_official": "SimLBR_released", "lbr_exact": "SimLBR_retrained", "sup": "supervised_probe", "recon": "reconstruction_probe",
        "rigid-dinov2": "RIGID_dinov2_pixel", "rigid-dinov2~norm": "RIGID_dinov2_norm", "rigid-dinov3": "RIGID_dinov3"}
DETERMINISTIC = {"rigid-dinov2", "rigid-dinov2~norm", "rigid-dinov3", "dv-tf"}
DVNAME = {"dv-single": "DejaVAE_I", "dv": "DejaVAE_P", "dv-nocf": "DejaVAE_P_noCF", "dv-single-nocf": "DejaVAE_I_noCF", "dv-tf": "DejaVAE_TF"}
SETS = {"genimage": ["dv2", "dv4", "dv"], "cham": ["dv2", "dv4", "dv"], "cham_m": ["dv2"], "cham_c": ["dv2"], "sb": ["dv2", "dv4", "dv"],
        "sb_crop": ["dv2", "dv"]}
TAG = {"dv2": "", "dv4": "_SD1decoder", "dv": "_finallayer"}   # dv: heads on the final-layer class token (App. E)
PUBLIC = {"genimage": "genimage", "cham": "chameleon", "cham_m": "chameleon_size_matched", "cham_c": "chameleon_processing_control",
          "sb": "synthbuster_geometry_matched", "sb_crop": "synthbuster_crop_matched", "sb_raw": "synthbuster_no_resave"}


def rel(p):
    for m in MARKERS:
        if m in p:
            return p[p.index(m):]
    return Path(p).name


def dv_scores(name, work):
    os.environ["DV_WORK"] = work; os.environ["DV_FEAT"] = "cls24" if work == "runs/dv" else "mid"
    for k in [k for k in sys.modules if k == "dejavae"]:
        del sys.modules[k]
    import dejavae as dv
    rows, idx = dv.rows_of(name)
    return rows, idx, dv.scores_for(name, rows, idx), dv.SETS[name][1]


def fmt(x):
    return f"{x:.6g}"


for name, works in SETS.items():
    cols, data = [], []
    rows, idx, S, ref = dv_scores(name, f"runs/{works[0]}")
    ix = np.array(idx); ref = Path(ref)
    def add(col, sc, dg):
        cols.append(col); data.append(sc[ix]); cols.append(col + "|degraded"); data.append(dg[ix])
    for m, pub in BASE.items():
        if (ref / f"scores_{m}_seed0.npy").exists():
            for s in ((0,) if m in DETERMINISTIC else (0, 1, 2)):
                add(f"{pub}@seed{s}", np.load(ref / f"scores_{m}_seed{s}.npy"), np.load(ref / f"degscores_{m}_seed{s}.npy"))
    for w in works:
        if w != works[0]:
            _, _, S, _ = dv_scores(name, f"runs/{w}")
        tag = TAG[w]
        for v, pub in DVNAME.items():
            if v in S and (w != "dv" or v in ("dv-single", "dv")):
                for s in ((0,) if v in DETERMINISTIC else (0, 1, 2)):
                    sc, dg = S[v][s]; add(f"{pub}{tag}@seed{s}", sc, dg)
    R = [rows[i] for i in idx]
    f = OUT / "scores" / f"{PUBLIC[name]}.csv.gz"
    with gzip.open(f, "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "split", "label", "source"] + cols)
        for j, r in enumerate(R):
            w.writerow([rel(r["path"]), r["split"], r["label"], r["source"]] + [fmt(d[j]) for d in data])
    print(f"{f.name}: {len(R)} rows, {len(cols)} score columns, {f.stat().st_size / 1e6:.1f} MB", flush=True)

# AEROBLADE (fixed stratified subsets): LPIPS distance per autoencoder; score = -min over autoencoders
for name, ref in {"genimage": "runs/p4s", "cham": "runs/cham_off", "cham_m": "runs/cham_m", "sb": "runs/sb", "sb_raw": "runs/sb_raw"}.items():
    a = Path(ref) / "aeroblade"
    if not (a / "idx.npy").exists():
        continue
    import dejavae as dv
    rows = ap.read_manifest(Path(dv.SETS[name][0])); idx = np.load(a / "idx.npy")
    D = {k: np.load(a / f"d_{k}.npz") for k in ("sd1", "sd2", "kd21")}
    f = OUT / "scores" / f"aeroblade_{PUBLIC[name]}.csv.gz"
    with gzip.open(f, "wt", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "split", "label", "source", "lpips_sd1", "lpips_sd2", "lpips_kandinsky21"])
        for j, i in enumerate(idx):
            r = rows[i]; w.writerow([rel(r["path"]), r["split"], r["label"], r["source"]] + [fmt(D[k]["d"][j]) for k in ("sd1", "sd2", "kd21")])
    print(f"{f.name}: {len(idx)} rows", flush=True)

# matching lists
with open(OUT / "matching" / "synthbuster_raise_split.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["raise_id", "split"]); seen = set()
    for r in ap.read_manifest(Path("runs/sb/manifest.csv")):
        if r["label"] == 0 and Path(r["path"]).stem not in seen:
            seen.add(Path(r["path"]).stem); w.writerow([Path(r["path"]).stem, r["split"]])
(OUT / "matching" / "synthbuster_excluded.txt").write_text(Path("runs/sb/excluded.txt").read_text())
from PIL import Image
for name, out in (("cham_m", "chameleon_size_matched_fakes.csv"), ("cham_c", "chameleon_processing_control_fakes.csv")):
    orig = {i: r for i, r in enumerate(ap.read_manifest(Path("runs/cham/manifest.csv")))}
    new = ap.read_manifest(Path(dv.SETS[name][0]))
    with open(OUT / "matching" / out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["original_image", "new_width", "new_height"])
        for i, r in enumerate(new):
            if r["label"] == 1:
                with Image.open(r["path"]) as im:
                    w.writerow([rel(orig[i]["path"]), im.size[0], im.size[1]])
    print(out, "written", flush=True)
