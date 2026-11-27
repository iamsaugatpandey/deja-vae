"""E5: Synthbuster (Bammey 2023; CC BY-NC-SA 4.0) + RAISE-1k reals, under a geometry-matched protocol.

Synthbuster fakes come at generator-specific sizes (GLIDE 256, SD1.x 512, DALL-E 1024, MJ 1360x896, Firefly up to
2688 px), while RAISE-1k reals are camera photographs (~4300-4900 px wide). Unmatched, size alone would separate the
classes (the Chameleon lesson). Each Synthbuster fake was generated from a prompt describing ONE RAISE-1k photo and is
named after it, so we match per image:

  for each generator g and RAISE id i:  real_{g,i} = RAISE_i centre-cropped to the aspect of fake_{g,i}, resized
  (bicubic, antialiased) to exactly fake_{g,i}'s size, saved losslessly (PNG).  Fakes are untouched (native output).

So for every generator, reals and fakes have identical size distributions, and the fakes keep their autoencoder fixed
point. All evaluation images then get the same JPEG q95 re-save inside each pipeline (as for GenImage).
Splits (seed 23, by RAISE id, 50/50): reals of calibration ids -> split "calib" (all 9 matched versions, pooled
threshold); reals of test ids -> "test" with source = generator; every fake -> "test" with source = generator.

    anchor/bin/python -I synthbuster_prep.py check            # fake-side stats, no RAISE needed
    anchor/bin/python -I synthbuster_prep.py make RAISE_DIR    # writes ~/anchor_data/synthbuster_real_matched + runs/sb/manifest.csv
RAISE_DIR holds the RAISE-1k images named <id>.TIF / .tif / .tiff / .png (from the RAISE CSV's TIFF column).
"""
import sys, csv
from pathlib import Path
import numpy as np

SB = Path.home() / "anchor_data" / "synthbuster" / "synthbuster"
OUT_IMG = Path.home() / "anchor_data" / "synthbuster_real_matched"
OUT = Path(__file__).parent / "runs" / "sb"
# Deviation from the pre-specified protocol (15 Nov 2026): RAISE r0bf7f938t is corrupt on the RAISE server itself
# (libtiff: strip 3962 truncated; a fresh download is byte-identical, size = server Content-Length). It is excluded
# together with its 9 paired fakes, so every generator keeps matched reals and fakes (999 ids). The calib/test split
# is still drawn over all 1,000 ids, so no other id changes split.
EXCLUDE = {"r0bf7f938t": "corrupt TIFF on the RAISE server (libtiff strip read error)"}
GENS = ["dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4",
        "stable-diffusion-2", "stable-diffusion-xl"]


def ids():
    return sorted(p.stem for p in (SB / "stable-diffusion-1-4").glob("*.png"))


def find_raise(rd, i):
    for ext in (".TIF", ".tif", ".tiff", ".TIFF", ".png", ".PNG", ".jpg"):
        p = rd / f"{i}{ext}"
        if p.exists():
            return p
    return None


def match(real, size):
    from PIL import Image
    tw, th = size; w, h = real.size; a = th / tw
    if h / w > a:
        ch = max(1, round(w * a)); t = (h - ch) // 2; real = real.crop((0, t, w, t + ch))
    else:
        cw = max(1, round(h / a)); l = (w - cw) // 2; real = real.crop((l, 0, l + cw, h))
    return real.resize((tw, th), Image.BICUBIC)


def cmd_check():
    I = ids(); print(len(I), "RAISE ids;", {g: len(list((SB / g).glob('*.png'))) for g in GENS})


def cmd_make(rd):
    from PIL import Image
    from tqdm import tqdm
    rd = Path(rd).expanduser(); I_all = ids(); I = [i for i in I_all if i not in EXCLUDE]
    missing = [i for i in I if find_raise(rd, i) is None]
    if missing:
        raise SystemExit(f"{len(missing)} RAISE images missing in {rd}, e.g. {missing[:3]}")
    rng = np.random.default_rng(23); perm = rng.permutation(len(I_all)); calib = {I_all[k] for k in perm[: len(I_all) // 2]}
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for i in tqdm(I, desc="match reals"):
        real = None
        for g in GENS:
            dst = OUT_IMG / g / f"{i}.png"
            if not dst.exists():
                if real is None:
                    with Image.open(find_raise(rd, i)) as im:
                        real = im.convert("RGB")
                with Image.open(SB / g / f"{i}.png") as f:
                    size = f.size
                dst.parent.mkdir(parents=True, exist_ok=True); match(real, size).save(dst, icc_profile=None)
            rows.append(dict(path=str(dst), label=0, split="calib" if i in calib else "test", source=g))
    for g in GENS:
        for i in I:
            rows.append(dict(path=str(SB / g / f"{i}.png"), label=1, split="test", source=g))
    (OUT / "excluded.txt").write_text("".join(f"{k}\t{v}\n" for k, v in EXCLUDE.items()))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"]); w.writeheader(); w.writerows(rows)
    n = np.array([(r["label"], r["split"]) for r in rows], dtype=object)
    print(f"wrote {OUT / 'manifest.csv'}: {sum(r['label'] == 0 and r['split'] == 'calib' for r in rows)} calib reals, "
          f"{sum(r['label'] == 0 and r['split'] == 'test' for r in rows)} test reals, {sum(r['label'] == 1 for r in rows)} fakes")


if __name__ == "__main__":
    {"check": lambda: cmd_check(), "make": lambda: cmd_make(sys.argv[2])}[sys.argv[1]]()
