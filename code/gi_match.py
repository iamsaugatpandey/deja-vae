"""Size-matched GenImage (revision, E4) and its processing-only control, on the E1 subset rows.

GenImage's real images are ImageNet JPEGs of many sizes; its generated images are PNG at a few fixed sizes per generator
(BigGAN 128, ADM/GLIDE/VQDM 256, SD 1.4/1.5 and Wukong 512, Midjourney 1024). Metadata alone separates the classes
(geometry AUROC 1.000, also after the JPEG q95 re-save). As for Chameleon (cham_match.py), each generated image gets the
geometry of a randomly drawn GenImage real (seed 19; RGB 4:2:0/4:4:4 JPEG reals from the calibration and test splits):
  gi_m (size-matched): centre crop to the real's aspect ratio, bicubic resize to exactly its width x height, saved as
       JPEG with THAT real's quantisation tables and chroma subsampling (so size, format and compression history match);
  gi_c (processing-only control): the same resize factor and the same JPEG tables, but the image keeps its own aspect
       ratio and full content.
Reals are untouched. Rows = the E1 GenImage subset (runs/p4s/aeroblade/idx.npy) in that order, so every version is
row-aligned with genimage_orig / genimage_q95 (all scores paired). Every pipeline then applies its q95 re-save as usual.
    anchor/bin/python -I gi_match.py make    -> ~/anchor_data/gi_{matched,ctrl}, runs/{gi_m,gi_c}/manifest.csv, idx_all.npy
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = {"gi_m": Path.home() / "anchor_data" / "gi_matched", "gi_c": Path.home() / "anchor_data" / "gi_ctrl"}


def jpeg_info(path):
    from PIL import Image, JpegImagePlugin
    with Image.open(path) as im:
        if im.format != "JPEG" or im.mode != "RGB":
            return None
        return im.size, {k: list(v) for k, v in im.quantization.items()}, JpegImagePlugin.get_sampling(im)


def make():
    from PIL import Image
    R = list(csv.DictReader(open(ROOT / "runs/p2/manifest.csv"))); I = np.load(ROOT / "runs/p4s/aeroblade/idx.npy")
    rows = [dict(R[i]) for i in I]
    for r in rows:
        if not Path(r["path"]).is_absolute():
            r["path"] = str(ROOT / r["path"])
    pool = [r["path"] if Path(r["path"]).is_absolute() else str(ROOT / r["path"])
            for r in R if r["label"] == "0" and r["split"] in ("calib", "test")]
    rng = np.random.default_rng(19); fk = [k for k, r in enumerate(rows) if r["label"] == "1"]
    out = {n: [dict(r) for r in rows] for n in OUT}; log = []
    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)
    for j, k in enumerate(fk):
        info = None
        while info is None:                                  # draw until an RGB JPEG real (skips grayscale / CMYK)
            info = jpeg_info(pool[rng.integers(0, len(pool))])
        (tw, th), qt, sub = info
        src = rows[k]["path"]; stem = f"{Path(src).stem}_{k}.jpg"
        with Image.open(src) as im0:
            im = im0.convert("RGB")
        w, h = im.size; a = th / tw
        if h / w > a:
            ch = max(1, round(w * a)); t = (h - ch) // 2; crop = im.crop((0, t, w, t + ch))
        else:
            cw = max(1, round(h / a)); l = (w - cw) // 2; crop = im.crop((l, 0, l + cw, h))
        s = tw / crop.size[0]
        dm, dc = OUT["gi_m"] / stem, OUT["gi_c"] / stem
        if not dm.exists():
            crop.resize((tw, th), Image.BICUBIC).save(dm, "JPEG", qtables=qt, subsampling=sub)
        if not dc.exists():
            im.resize((max(8, round(w * s)), max(8, round(h * s))), Image.BICUBIC).save(dc, "JPEG", qtables=qt, subsampling=sub)
        out["gi_m"][k]["path"] = str(dm); out["gi_c"][k]["path"] = str(dc)
        log.append([w, h, crop.size[0], crop.size[1], tw, th])
        if j % 250 == 0:
            print(f"matched {j}/{len(fk)}", flush=True)
    for n, rr in out.items():
        d = ROOT / "runs" / n; d.mkdir(parents=True, exist_ok=True)
        with open(d / "manifest.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"]); wr.writeheader()
            wr.writerows([{k: r[k] for k in ("path", "label", "split", "source")} for r in rr])
        np.save(d / "idx_all.npy", np.arange(len(rr)))
    L = np.array(log); sc = L[:, 4] / L[:, 2]
    np.save(ROOT / "runs/gi_m/match_log.npy", L)
    print(f"{len(fk)} generated images matched; resize factor median {np.median(sc):.2f} (upscaled {100 * (sc > 1).mean():.0f}%), "
          f"area kept by the crop median {np.median(L[:, 2] * L[:, 3] / (L[:, 0] * L[:, 1])):.2f}", flush=True)


if __name__ == "__main__":
    {"make": make}[sys.argv[1]]()
