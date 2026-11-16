"""RIGID baseline (He, Chen & Ho, TMLR): training-free. An image x is scored by the cosine similarity
between foundation-model embeddings of x and x + lambda * delta, delta ~ N(0, I), lambda = 0.05, one
draw; LOWER similarity = more likely generated (real images are more robust to the perturbation).
Paper backbone: DINOv2 ViT-L/14 (CLS embedding; preprocessing unspecified -> standard DINOv2
transform: resize 256, centre crop 224, ImageNet normalisation; noise added in [0, 1] pixel space).
We also run the same rule on our DINOv3 ViT-L/16 features (SimLBR preprocessing) for backbone parity.

Rows and image handling follow the evaluation folders exactly (every calib/test row):
  runs/p4s      GenImage, JPEG q95 re-save (p2 protocol), degraded copies of test reals
  runs/cham_off Chameleon, no re-save
Writes scores_rigid-<bb>_seed{0,1,2}.npy and degscores_... (one deterministic score, copied per seed
so paired comparisons with 3-seed methods work), so `anchor_pilot.py eval` and the reports pick them up.

    anchor/bin/python rigid.py --work runs/p4s --test_jpeg 95 --backbone dinov2|dinov3
    anchor/bin/python rigid.py --work runs/cham_off --backbone dinov2|dinov3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap

LAM = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True); p.add_argument("--test_jpeg", type=int, default=0)
    p.add_argument("--backbone", choices=["dinov2", "dinov3"], required=True); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--idx", default="", help="score only these rows (.npy of row indices), e.g. a sanity subset")
    p.add_argument("--noise_space", choices=["pixel", "norm"], default="pixel",
                   help="pixel: noise in [0,1] pixel space (default); norm: noise added after normalisation")
    p.add_argument("--suffix", default="", help="appended to the score tag, e.g. ~raw-sub")
    p.add_argument("--seed_from", default="", help="folder with the same rows' scores: copy label-0 rows, score only fakes")
    a = p.parse_args()
    import torch
    from PIL import ImageFilter
    from transformers import AutoModel
    W = Path(a.work); rows = ap.read_manifest(W / "manifest.csv"); N = len(rows)
    O = Path(a.idx).parent if a.idx else W          # subset runs write next to their index file, not into W
    tag = f"rigid-{a.backbone}" + ("~norm" if a.noise_space == "norm" else "") + a.suffix
    if (O / f"scores_{tag}_seed2.npy").exists():
        print(f"{tag} already scored in {W}"); return
    dev = ap.pick_device()
    if a.backbone == "dinov2":
        model = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
        mean, std = ap.IMNET_MEAN, ap.IMNET_STD

        def prep(im):
            from PIL import Image
            w, h = im.size; s = 256 / min(w, h)
            im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
            w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
            return im.crop((l, t, l + 224, t + 224))
    else:
        model = AutoModel.from_pretrained(ap.DINO_ID).to(dev).eval()
        mean, std = ap.SIMLBR_MEAN, ap.SIMLBR_STD
        prep = None   # SimLBR squash resize, applied in load
    M = torch.tensor(mean, device=dev).view(1, 3, 1, 1); S = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    gen = torch.Generator(device="cpu").manual_seed(0)

    def load(path):
        if a.backbone == "dinov3":
            im = ap.load_rgb(path, 256, "simlbr")
        else:
            from PIL import Image
            im = prep(Image.open(path).convert("RGB"))
        return im

    rng = np.random.default_rng(0)                      # degraded copies: same draws as `features` (seed 0)
    q = np.where(rng.random(N) < 1.0, rng.integers(70, 101, N), 0)
    sig = rng.uniform(0.5, 1.5, N)
    idx = [i for i, r in enumerate(rows) if r["split"] != "train"]
    if a.idx:
        keep = set(np.load(a.idx).tolist()); idx = [i for i in idx if i in keep]
    sc = np.zeros(N, np.float32); dg = np.zeros(N, np.float32)
    if a.seed_from:                                     # reals identical to that folder (same row order): reuse them
        sf = Path(a.seed_from); keep0 = np.array([r["label"] == 0 for r in rows])
        sc[keep0] = np.load(sf / f"scores_{tag}_seed0.npy")[keep0]; dg[keep0] = np.load(sf / f"degscores_{tag}_seed0.npy")[keep0]
        idx = [i for i in idx if rows[i]["label"] == 1]

    def sims(ims):
        x = torch.from_numpy(np.stack([np.asarray(im, np.float32) / 255 for im in ims])).permute(0, 3, 1, 2).contiguous()
        noise = LAM * torch.randn(x.shape, generator=gen)
        with torch.no_grad():
            e = lambda t: model(pixel_values=t.to(dev)).last_hidden_state[:, 0]
            xs = (x.to(dev) - M) / S
            if a.noise_space == "pixel":
                xns = ((x + noise).clamp(0, 1).to(dev) - M) / S      # noise in [0, 1] pixel space
            else:
                xns = xs + noise.to(dev)                             # noise on the normalised input
            return torch.nn.functional.cosine_similarity(e(xs), e(xns), dim=1).float().cpu().numpy()

    t0 = time.time()
    for s in range(0, len(idx), a.batch):
        b = idx[s:s + a.batch]
        ims = [load(rows[i]["path"]) for i in b]
        if a.test_jpeg:
            ims = [ap.jpeg(im, a.test_jpeg) for im in ims]
        sc[b] = -sims(ims)                               # higher = more likely generated
        dpos = [k for k, i in enumerate(b) if rows[i]["label"] == 0 and rows[i]["split"] == "test"]
        if dpos:
            dims = [load(rows[b[k]]["path"]).filter(ImageFilter.GaussianBlur(float(sig[b[k]]))) for k in dpos]
            if a.test_jpeg:
                dims = [ap.jpeg(im, a.test_jpeg) for im in dims]
            dg[[b[k] for k in dpos]] = -sims(dims)
        if (s // a.batch) % 50 == 0:
            el = time.time() - t0; done = s + len(b)
            print(f"{tag} {W.name}: {done}/{len(idx)}  {done / el:.1f} img/s  eta {(len(idx) - done) / max(done / el, 1e-6) / 60:.0f} min", flush=True)
    for seed in (0, 1, 2):
        np.save(O / f"scores_{tag}_seed{seed}.npy", sc); np.save(O / f"degscores_{tag}_seed{seed}.npy", dg)
    print(f"{tag} {W.name} DONE in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
