"""EvalGEN (DDA, Chen et al. 2025: FLUX, GoT, Infinity, NOVA, OmniGen; generated images only) + matched RAISE-1k reals.

EvalGEN ships generated images only, all 1024 x 1024 JPEG with PIL's standard quality-96 tables (4:2:0; no EXIF/ICC;
checked on every generator). We bring real photographs under the same file conditions, so that size, format and
compression history cannot separate the classes:
  reals  = the 999 RAISE-1k photographs of runs/sb (same ids, same calibration/test split, seed 23), each
           eg:       centre-cropped to a square and bicubic-downscaled to 1024 x 1024  (as synthbuster_prep.match)
           eg_crop:  native-resolution centre crop of 1024 x 1024 (no resampling)     (as synthbuster_crop)
           then saved as JPEG quality 96 by PIL (same tables and subsampling as EvalGEN), no EXIF, no ICC.
  fakes  = 300 images per generator, drawn with seed 29 from the files that decode (1,500 in total), untouched.
Every image then gets the pipelines' JPEG q95 re-save, so both classes are compressed twice with the same tables.
Content is NOT matched: EvalGEN prompts are GenEval object prompts, RAISE are camera scenes. TPR here is the share of
2024-25 generated images scored above the 95th percentile of matched real photographs.
    anchor/bin/python -I evalgen_prep.py make     -> ~/anchor_data/evalgen_real_{matched,cropped}, runs/{eg,eg_crop}/manifest.csv
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EG = Path.home() / "anchor_data" / "bench" / "evalgen" / "EvalGEN"
GENS = ["Flux", "GoT", "Infinity", "NOVA", "OmniGen"]
N_FAKE, SIZE = 300, 1024
OUT = {"eg": Path.home() / "anchor_data" / "evalgen_real_matched", "eg_crop": Path.home() / "anchor_data" / "evalgen_real_cropped"}


def find_raise(i):
    for ext in (".TIF", ".tif", ".tiff", ".TIFF"):
        p = Path.home() / "anchor_data" / "raise" / "tiff" / f"{i}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(i)


def downscaled(im):
    from PIL import Image
    w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.BICUBIC)


def cropped(im):
    w, h = im.size; l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return im.crop((l, t, l + SIZE, t + SIZE))


def cmd_make():
    from PIL import Image
    sb = list(csv.DictReader(open(ROOT / "runs/sb/manifest.csv")))
    split = {}
    for r in sb:
        if r["label"] == "0":
            split[Path(r["path"]).stem] = r["split"]
    ids = sorted(split)
    rng = np.random.default_rng(29); fakes = []
    for g in GENS:
        files = sorted(p for p in (EG / g).glob("*.jpg"))
        ok = []
        for k in rng.permutation(len(files)):
            try:
                with Image.open(files[k]) as im:
                    im.load()
                    if im.size == (SIZE, SIZE):
                        ok.append(files[k])
            except Exception:
                continue
            if len(ok) == N_FAKE:
                break
        fakes += [(str(p), g) for p in sorted(ok)]
    for name, out in OUT.items():
        out.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(ids):
        todo = {name: OUT[name] / f"{i}.jpg" for name in OUT if not (OUT[name] / f"{i}.jpg").exists()}
        if todo:
            with Image.open(find_raise(i)) as im:
                real = im.convert("RGB")
            for name, dst in todo.items():
                (downscaled if name == "eg" else cropped)(real).save(dst, format="JPEG", quality=96)
        if n % 100 == 0:
            print(f"reals {n}/{len(ids)}", flush=True)
    for name in OUT:
        d = ROOT / "runs" / name; d.mkdir(parents=True, exist_ok=True)
        rows = [dict(path=str(OUT[name] / f"{i}.jpg"), label=0, split=split[i], source="raise") for i in ids]
        rows += [dict(path=p, label=1, split="test", source=g) for p, g in fakes]
        with open(d / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"]); w.writeheader(); w.writerows(rows)
        np.save(d / "idx_all.npy", np.arange(len(rows)))
        print(f"{name}: {sum(r['split'] == 'calib' for r in rows)} calib reals, "
              f"{sum(r['label'] == 0 and r['split'] == 'test' for r in rows)} test reals, {len(fakes)} fakes", flush=True)


if __name__ == "__main__":
    {"make": cmd_make}[sys.argv[1]]()
