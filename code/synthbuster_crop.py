"""Synthbuster with CROP-matched reals (reviewer control for the resampling asymmetry of synthbuster_prep.py).
synthbuster_prep.py matches each RAISE photograph to its paired fake's size by centre-crop to the aspect ratio + bicubic
DOWNSCALE (about 2-19x). The fakes are never resampled, so "was strongly downscaled" could separate the classes.
Here each real is instead a NATIVE-resolution centre crop of the RAISE photograph at exactly the fake's size: no
resampling at all. Same RAISE ids, same calibration/test split, same fakes, same exclusion as runs/sb.
    anchor/bin/python -I synthbuster_crop.py make RAISE_TIFF_DIR   # -> ~/anchor_data/synthbuster_real_cropped, runs/sb_crop/manifest.csv
"""
import sys, csv
from pathlib import Path

OUT_IMG = Path.home() / "anchor_data" / "synthbuster_real_cropped"
SRC = Path("runs/sb/manifest.csv"); DST = Path("runs/sb_crop"); DST.mkdir(parents=True, exist_ok=True)


def find_raise(rd, i):
    for ext in (".TIF", ".tif", ".tiff", ".TIFF"):
        if (rd / f"{i}{ext}").exists():
            return rd / f"{i}{ext}"
    raise FileNotFoundError(i)


def cmd_make(rd):
    from PIL import Image
    rd = Path(rd).expanduser()
    rows = list(csv.DictReader(open(SRC)))
    need = {}                                   # raise id -> [(generator, (w, h))]
    for r in rows:
        if r["label"] == "0":
            with Image.open(r["path"]) as im:
                need.setdefault(Path(r["path"]).stem, []).append((r["source"], im.size))
    small = 0
    for n, (i, todo) in enumerate(sorted(need.items())):
        with Image.open(find_raise(rd, i)) as im:
            im = im.convert("RGB"); W, H = im.size
            for g, (w, h) in todo:
                if w > W or h > H:              # fake larger than the photo in one side: rotate the photo if that fits
                    small += 1
                    src = im.transpose(Image.Transpose.ROTATE_90) if (h <= W and w <= H) else im
                else:
                    src = im
                SW, SH = src.size; cw, ch = min(w, SW), min(h, SH)
                l, t = (SW - cw) // 2, (SH - ch) // 2
                dst = OUT_IMG / g / f"{i}.png"; dst.parent.mkdir(parents=True, exist_ok=True)
                src.crop((l, t, l + cw, t + ch)).save(dst, icc_profile=None)
        if n % 50 == 0:
            print(f"{n}/{len(need)}", flush=True)
    with open(DST / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "split", "source"])
        for r in rows:
            p = str(OUT_IMG / r["source"] / Path(r["path"]).name) if r["label"] == "0" else r["path"]
            w.writerow([p, r["label"], r["split"], r["source"]])
    print(f"done: {len(need)} RAISE ids; {small} crops needed rotation or were clipped", flush=True)


if __name__ == "__main__":
    {"make": lambda: cmd_make(sys.argv[2])}[sys.argv[1]]()
