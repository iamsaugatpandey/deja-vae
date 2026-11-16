#!/usr/bin/env python3
"""
ANCHOR pilot (Stage 1): does mixing toward each image's OWN reconstruction help
beyond training on reconstructions alone?

Five training conditions share one frozen backbone (DINOv3 ViT-L/16, CLS token),
one MLP head, the same real images and the same preprocessing. Only the negatives
(practice fakes) differ:

  A  sup                  actual generated images
  B  lbr                  SimLBR: a*h(real) + (1-a)*h(actual fake), a ~ U(0.5, 0.8)
  C  recon                h(reconstruction of the real), unmixed
  D  mix_unpaired         (1-b)*h(real_i) + b*h(recon of ANOTHER real_j)
     mix_unpaired_matched same, displacement rescaled to the size of real_i's own
  E  mix_aligned          (1-b)*h(real_i) + b*h(recon of real_i), b ~ U(b_min, b_max)

The decisive comparison is E vs C and E vs D (matched). Beating B alone does not
show that aligned mixing is responsible.

Controls built in:
  * degradation-matched reals: each training real also gets a benign copy (Gaussian
    blur tuned to the same PSNR as its reconstruction, same JPEG re-save), which
    can be added to training as REAL (--degraded_reals) and is always evaluated:
    a detector that flags these is detecting detail loss, not generation.
  * every method is calibrated on the SAME held-out real images (split conformal),
    and compared by fake-detection rate at matched false-alarm targets.
  * paired bootstrap 95% CIs for E - C and E - D.

Zero-fake levels: C, D, E use no generated image for fitting (Z1). With the default,
fixed-in-advance hyperparameters they also use none for model selection (Z2).
A and B use actual fakes by design.

Subcommands: manifest | bench | features | train | eval | status   (see README.md)
Install (Mac): pip install torch torchvision "transformers>=4.56" diffusers accelerate scikit-learn pillow tqdm pandas
(DINOv3 needs transformers >= 4.56; older versions cannot load it.)
"""
import argparse, csv, io, json, math, os, random, re, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before torch is imported

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}   # compared lower-case
DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VAE_ID = "stabilityai/sd-vae-ft-mse"
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SIMLBR_TAGS = ("lbr", "lbr_exact", "simlbr_official")   # our B, the released recipe, the released head
SIMLBR_MEAN, SIMLBR_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)   # released SimLBR heads (mvrl/SimLBR)
METHODS = ["sup", "lbr", "lbr_exact", "recon", "mix_unpaired", "mix_unpaired_matched", "mix_aligned"]
LETTER = {"sup": "A", "lbr": "B", "lbr_exact": "B=", "simlbr_official": "B-off", "recon": "C",
          "mix_unpaired": "D", "mix_unpaired_matched": "D*", "mix_aligned": "E"}


# ----------------------------------------------------------------------------- utils
def base_method(tag):
    """'mix_aligned+deg@sd-flux' -> 'mix_aligned'"""
    return re.split(r"[+@~]", tag)[0]


def pick_device(pref="auto"):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_manifest(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def load_rgb(path, size, preproc="crop"):
    """crop (default): shorter side -> size, centre crop size x size.
    simlbr: resize to size x size without crop, exactly as the released SimLBR transform."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if preproc == "simlbr":
        import torch
        from torchvision.transforms import v2
        t = v2.Resize((size, size), antialias=True)(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
        return Image.fromarray(t.permute(1, 2, 0).numpy())
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def jpeg(im, q):
    from PIL import Image
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def psnr(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def blur_to_psnr(im, target, lo=0.05, hi=4.0, iters=12):
    """Benign Gaussian blur whose PSNR to the original matches `target` (binary search on sigma)."""
    from PIL import ImageFilter
    best, best_err = im, float("inf")
    for _ in range(iters):
        mid = (lo + hi) / 2
        cand = im.filter(ImageFilter.GaussianBlur(mid))
        p = psnr(im, cand)
        if abs(p - target) < best_err:   # keep the closest match, not just the last probe
            best, best_err = cand, abs(p - target)
        if p > target:   # not degraded enough
            lo = mid
        else:
            hi = mid
    return best


# ----------------------------------------------------------------------------- 1. manifest
def cmd_manifest(a):
    """--entry DIR,real|fake,train|calib|test,SOURCE[,MAX]"""
    rng = random.Random(a.seed)
    rows, seen_real, n_dup = [], set(), 0
    for e in a.entry:
        parts = e.split(",")
        if len(parts) < 4 or parts[1] not in ("real", "fake") or parts[2] not in ("train", "calib", "test"):
            raise SystemExit(f"bad --entry {e!r}: expected DIR,real|fake,train|calib|test,SOURCE[,MAX]")
        d, lab, split, src = parts[:4]
        mx = int(parts[4]) if len(parts) > 4 else None
        files = sorted(p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        if not files:
            print(f"warning: no images under {d}")
        kept = 0
        for p in files:
            if mx and kept >= mx:
                break
            if lab == "real":
                # GenImage reuses the same ImageNet photos in several generators' nature folders:
                # keep each real photo once so it cannot sit in train/calib and test at the same time.
                key = (p.name, p.stat().st_size)
                if key in seen_real:
                    n_dup += 1
                    continue
                seen_real.add(key)
            rows.append(dict(path=str(p), label=1 if lab == "fake" else 0, split=split, source=src))
            kept += 1
    if n_dup:
        print(f"dropped {n_dup} duplicate real images (same file name and size under several entries)")
    if a.calib_frac > 0:
        test_real = [r for r in rows if r["split"] == "test" and r["label"] == 0]
        rng.shuffle(test_real)
        for r in test_real[: int(len(test_real) * a.calib_frac)]:
            r["split"] = "calib"
    out = Path(a.work) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"])
        w.writeheader(); w.writerows(rows)
    summary = {}
    for r in rows:
        k = (r["split"], r["source"], "fake" if r["label"] else "real")
        summary[k] = summary.get(k, 0) + 1
    for k in sorted(summary):
        print(f"{k[0]:6s} {k[1]:20s} {k[2]:4s} {summary[k]:7d}")
    print(f"wrote {out}  ({len(rows)} images)")
    print("NOTE: keep everything derived from one source (same photo, chart table, paper, patient) in ONE split.")


# ----------------------------------------------------------------------------- models
def load_autoencoder(spec, device):
    """spec = REPO or REPO:SUBFOLDER. The class (AutoencoderKL, VQModel, AutoencoderDC, ...) is read
    from the model's own config, so any diffusers autoencoder family works."""
    import diffusers
    repo, sub = (spec.split(":", 1) + [None])[:2]
    cfg = diffusers.AutoencoderKL.load_config(repo, subfolder=sub)
    cls = getattr(diffusers, cfg["_class_name"])
    return cls.from_pretrained(repo, subfolder=sub).to(device).eval()


class Encoders:
    def __init__(self, device, need_vae=True, backbone=DINO_ID, vae=VAE_ID, norm="imagenet"):
        import torch
        from transformers import AutoModel
        self.torch, self.device = torch, device
        try:
            self.dino = AutoModel.from_pretrained(backbone).to(device).eval()
        except OSError as err:
            raise SystemExit(f"could not load {backbone}: {err}\n"
                             "DINOv3 weights are gated: request access on its Hugging Face page, then run "
                             "`hf auth login` (or `huggingface-cli login`).")
        self.dim = int(self.dino.config.hidden_size)
        self.vae = None
        if need_vae:
            self.vae = load_autoencoder(vae, device)
        mean, std = (SIMLBR_MEAN, SIMLBR_STD) if norm == "simlbr" else (IMNET_MEAN, IMNET_STD)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def to_tensor(self, ims):
        arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in ims])
        # .contiguous(): the NHWC->NCHW permute leaves a channels-last view; on MPS that layout
        # propagates into the VAE latents and makes decoding ~10x slower.
        return self.torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(self.device)

    def cls(self, ims):
        with self.torch.no_grad():
            x = (self.to_tensor(ims) - self.mean) / self.std
            out = self.dino(pixel_values=x).last_hidden_state  # [CLS, registers..., patches]
            return out[:, 0].float().cpu().numpy()

    def reconstruct(self, ims, vae_batch=4):
        """Autoencoder encode -> decode: content-matched reconstruction (also loses some detail).
        Runs in sub-batches: at 256 px the decoder's activations are ~0.5 GB per tensor at batch 16,
        which thrashes a 16 GB Mac (measured 0.4-0.6 img/s at 16 vs a steady 4.5 img/s at 4-8)."""
        from PIL import Image
        t = self.torch
        out = []
        with t.no_grad():
            for s in range(0, len(ims), vae_batch):
                x = self.to_tensor(ims[s:s + vae_batch]) * 2 - 1
                e = self.vae.encode(x)
                if hasattr(e, "latent_dist"):      # KL autoencoders (SD, SDXL, FLUX)
                    z = e.latent_dist.mode()
                elif hasattr(e, "latents"):        # VQ models: decode() quantizes
                    z = e.latents
                else:                              # deep-compression AE
                    z = e.latent
                y = self.vae.decode(z.contiguous()).sample.clamp(-1, 1)
                y = ((y + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                out += [Image.fromarray(v) for v in y]
        return out


# ----------------------------------------------------------------------------- 2. bench
def cmd_bench(a):
    from PIL import Image
    dev = pick_device(a.device)
    enc = Encoders(dev, backbone=a.backbone, vae=a.vae)
    ims = [Image.fromarray(np.random.randint(0, 255, (a.size, a.size, 3), dtype=np.uint8)) for _ in range(a.batch)]
    for name, fn in [("DINOv3 CLS features", enc.cls), ("VAE reconstruction", lambda x: enc.reconstruct(x, a.vae_batch))]:
        fn(ims)
        if dev.type == "mps":
            enc.torch.mps.synchronize()
        t0 = time.time()
        for _ in range(3):
            fn(ims)
        if dev.type == "mps":
            enc.torch.mps.synchronize()
        ips = 3 * a.batch / (time.time() - t0)
        print(f"{name:22s} {ips:7.1f} images/s on {dev}  ->  10k images in {10000 / ips / 60:5.1f} min")


# ----------------------------------------------------------------------------- 3. features
def cmd_features(a):
    """Caches, per manifest row:
       cls.npy           features of the image itself
       cls_recon.npy     features of its reconstruction   (train reals only)
       cls_degraded.npy  features of a benign degraded copy (all reals):
                         train reals: blur matched to the reconstruction's PSNR
                         calib/test reals: random blur sigma in [0.5, 1.5]
       All members of a train trio get the same JPEG re-save (or none), and training fakes
       get the same random re-save distribution, so file format is not a shortcut for A/B."""
    from tqdm import tqdm
    dev = pick_device(a.device)
    W = Path(a.work)
    rows = read_manifest(W / "manifest.csv")
    enc = Encoders(dev, need_vae=not a.skip_train, backbone=a.backbone, vae=a.vae,
                   norm="simlbr" if a.preproc == "simlbr" else "imagenet")
    rng = np.random.default_rng(a.seed)
    N, D = len(rows), enc.dim
    meta_path = W / "features_meta.json"
    meta = dict(n_rows=N, dim=D, backbone=a.backbone, vae=a.vae, size=a.size, seed=a.seed,
                jpeg_prob=a.jpeg_prob, jpeg_min=a.jpeg_min)
    if a.test_jpeg:   # only recorded when used, so folders made before this option still resume
        meta["test_jpeg"] = a.test_jpeg
    if a.preproc != "crop":
        meta["preproc"] = a.preproc
    if a.skip_train:
        meta["skip_train"] = True
    resume = (W / "cls.npy").exists()
    if resume:
        old = json.load(open(meta_path)) if meta_path.exists() else {}
        if any(old.get(k) != v for k, v in meta.items()):
            raise SystemExit(f"{W} already holds features made with different settings or a different manifest:\n"
                             f"  cached {old}\n  now    {meta}\n"
                             "Use a new --work folder, or delete cls*.npy, features_done.json and features_meta.json.")
    else:
        json.dump(meta, open(meta_path, "w"), indent=1)
    mode = "r+" if resume else "w+"
    F = np.lib.format.open_memmap(W / "cls.npy", mode=mode, dtype=np.float16, shape=(N, D))
    R = np.lib.format.open_memmap(W / "cls_recon.npy", mode=mode, dtype=np.float16, shape=(N, D))
    G = np.lib.format.open_memmap(W / "cls_degraded.npy", mode=mode, dtype=np.float16, shape=(N, D))
    done_path = W / "features_done.json"
    done = set(json.load(open(done_path))) if done_path.exists() else set()
    q = np.where(rng.random(N) < a.jpeg_prob, rng.integers(a.jpeg_min, 101, N), 0)
    sig = rng.uniform(0.5, 1.5, N)
    psnr_log = []
    if a.skip_train:   # evaluation-only copy (e.g. for the released SimLBR head): train rows stay zero
        done.update(i for i, r in enumerate(rows) if r["split"] == "train")
    todo = [i for i in range(N) if i not in done]
    t0 = time.time()
    from PIL import ImageFilter
    for s in tqdm(range(0, len(todo), a.batch), desc="features"):
        idx = todo[s: s + a.batch]
        ims = [load_rgb(rows[i]["path"], a.size, a.preproc) for i in idx]
        is_tr = [rows[i]["split"] == "train" and rows[i]["label"] == 0 for i in idx]
        is_real = [rows[i]["label"] == 0 for i in idx]
        sel = [k for k, f in enumerate(is_tr) if f]
        recs = dict(zip(sel, enc.reconstruct([ims[k] for k in sel], a.vae_batch))) if sel else {}
        main, rec_ims, rec_pos, deg_ims, deg_pos = [], [], [], [], []
        for k, i in enumerate(idx):
            im = ims[k]
            if k in recs:
                rc = recs[k]
                p = psnr(im, rc); psnr_log.append(p)
                dg = blur_to_psnr(im, p)
                if q[i] > 0:
                    im, rc, dg = jpeg(im, q[i]), jpeg(rc, q[i]), jpeg(dg, q[i])
                rec_ims.append(rc); rec_pos.append(i)
                deg_ims.append(dg); deg_pos.append(i)
            elif is_real[k]:
                dg = im.filter(ImageFilter.GaussianBlur(float(sig[i])))
                if a.test_jpeg:   # format-aligned evaluation: every calib/test image gets the same re-save
                    im, dg = jpeg(im, a.test_jpeg), jpeg(dg, a.test_jpeg)
                deg_ims.append(dg); deg_pos.append(i)
            elif rows[i]["split"] == "train" and q[i] > 0:   # training fakes (A, B): same re-save distribution
                im = jpeg(im, q[i])
            elif rows[i]["split"] != "train" and a.test_jpeg:
                im = jpeg(im, a.test_jpeg)
            main.append(im)
        F[idx] = enc.cls(main).astype(np.float16)
        if rec_ims:
            R[rec_pos] = enc.cls(rec_ims).astype(np.float16)
        if deg_ims:
            G[deg_pos] = enc.cls(deg_ims).astype(np.float16)
        done.update(idx)
        if (s // a.batch) % 10 == 0:   # checkpoint (also what `status` reads)
            F.flush(); R.flush(); G.flush(); json.dump(sorted(done), open(done_path, "w"))
        if s == 0:
            print(f"first batch: {len(idx) / (time.time() - t0):.1f} images/s (incl. loading and reconstructions)", flush=True)
    F.flush(); R.flush(); G.flush(); json.dump(sorted(done), open(done_path, "w"))
    if psnr_log:
        print(f"reconstruction PSNR: median {np.median(psnr_log):.1f} dB (degraded reals matched to it)")
    print(f"cached {len(done)}/{N} rows in {W}")


def cmd_recon(a):
    """Adds one autoencoder to the reconstruction bank (training reals only):
       cls_recon_<name>.npy     features of the reconstruction
       cls_degraded_<name>.npy  features of a benign blur matched to THAT reconstruction's PSNR
       Uses the same image size, seed and JPEG re-save draws as `features` (read from
       features_meta.json), so each trio stays aligned with the cached features of the original."""
    from tqdm import tqdm
    W = Path(a.work)
    meta = json.load(open(W / "features_meta.json"))
    rows = read_manifest(W / "manifest.csv")
    N = len(rows)
    if meta["n_rows"] != N:
        raise SystemExit("manifest changed since `features`")
    rng = np.random.default_rng(meta["seed"])                       # replicate the draws of cmd_features
    q = np.where(rng.random(N) < meta["jpeg_prob"], rng.integers(meta["jpeg_min"], 101, N), 0)
    dev = pick_device(a.device)
    enc = Encoders(dev, backbone=meta["backbone"], vae=a.vae)
    D = enc.dim
    rpath, gpath = W / f"cls_recon_{a.name}.npy", W / f"cls_degraded_{a.name}.npy"
    done_path, mpath = W / f"recon_{a.name}_done.json", W / f"recon_{a.name}_meta.json"
    rmeta = dict(vae=a.vae, n_rows=N)
    if rpath.exists():
        if json.load(open(mpath)) != rmeta:
            raise SystemExit(f"{rpath} was made with different settings: delete it or use another --name")
        mode = "r+"
    else:
        json.dump(rmeta, open(mpath, "w")); mode = "w+"
    R = np.lib.format.open_memmap(rpath, mode=mode, dtype=np.float16, shape=(N, D))
    G = np.lib.format.open_memmap(gpath, mode=mode, dtype=np.float16, shape=(N, D))
    done = set(json.load(open(done_path))) if done_path.exists() else set()
    tr = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0 and i not in done]
    psnr_log = []
    for s in tqdm(range(0, len(tr), a.batch), desc=f"recon {a.name}"):
        idx = tr[s:s + a.batch]
        ims = [load_rgb(rows[i]["path"], meta["size"]) for i in idx]
        recs = enc.reconstruct(ims, a.vae_batch)
        rc_out, dg_out = [], []
        for im, rc, i in zip(ims, recs, idx):
            p = psnr(im, rc); psnr_log.append(p)
            dg = blur_to_psnr(im, p)
            if q[i] > 0:
                rc, dg = jpeg(rc, q[i]), jpeg(dg, q[i])
            rc_out.append(rc); dg_out.append(dg)
        R[idx] = enc.cls(rc_out).astype(np.float16)
        G[idx] = enc.cls(dg_out).astype(np.float16)
        done.update(idx)
        if (s // a.batch) % 10 == 0:
            R.flush(); G.flush(); json.dump(sorted(done), open(done_path, "w"))
    R.flush(); G.flush(); json.dump(sorted(done), open(done_path, "w"))
    if psnr_log:
        print(f"[{a.name}] reconstruction PSNR: median {np.median(psnr_log):.1f} dB")
    print(f"[{a.name}] cached reconstructions for {len(done)} training reals")


def load_bank(W, names):
    """[(R, G)] for each bank member; 'sd' (or 'default') is the one made by `features`."""
    out = []
    for n in names:
        if n in ("sd", "default"):
            out.append((np.load(W / "cls_recon.npy", mmap_mode="r"), np.load(W / "cls_degraded.npy", mmap_mode="r")))
        else:
            p = W / f"cls_recon_{n}.npy"
            if not p.exists():
                raise SystemExit(f"bank member {n!r} missing: run `recon --name {n} --vae ...` first")
            out.append((np.load(p, mmap_mode="r"), np.load(W / f"cls_degraded_{n}.npy", mmap_mode="r")))
    return out


# ----------------------------------------------------------------------------- 4. training sets (numpy, testable)
def make_training_set(F, R, G, rows, method, rng, cfg):
    """One epoch of (X, y). Mixtures are re-drawn every epoch, as in SimLBR.
    C, D, D*, E all use len(train reals) negatives from the same reconstruction bank."""
    tr_real = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0])
    tr_fake = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 1])
    Xr = F[tr_real].astype(np.float32)
    Xs, ys = [Xr], [np.zeros(len(Xr))]
    # reconstruction bank: K autoencoders, shape [K, n_real, D]; K = 1 is the single default VAE
    if "_bank" not in cfg:
        bank = cfg.get("bank") or [(R, G)]
        cfg["_bank"] = (np.stack([b[0][tr_real].astype(np.float32) for b in bank]),
                        np.stack([b[1][tr_real].astype(np.float32) for b in bank]))
    RB, GB = cfg["_bank"]
    K, nr = RB.shape[0], RB.shape[1]
    pick = (lambda: rng.integers(0, K, nr)) if K > 1 else (lambda: np.zeros(nr, dtype=int))
    if cfg.get("degraded_reals"):
        Xg = GB[pick(), np.arange(nr)]                       # each real's degraded copy, from a random bank member
        Xs.append(Xg); ys.append(np.zeros(len(Xg)))
    uses_fakes = method in ("sup", "lbr") or cfg.get("with_fakes")
    def simlbr_blends():
        """Released SimLBR recipe: every training fake is REPLACED by alpha*real + (1-alpha)*fake,
        alpha ~ U(0.5, 0.8), with a fresh random real; raw fakes are not shown."""
        if not len(tr_fake):
            raise SystemExit(f"{method} needs training fakes in the manifest")
        n = len(tr_fake)
        ri = rng.integers(0, len(Xr), n)
        al = rng.uniform(cfg["a_min"], cfg["a_max"], (n, 1)).astype(np.float32)
        return al * Xr[ri] + (1 - al) * F[tr_fake].astype(np.float32)
    if method == "lbr_exact":
        Z = simlbr_blends()
        Xs.append(Z); ys.append(np.ones(len(Z)))
        X = np.concatenate(Xs); y = np.concatenate(ys)
        p = rng.permutation(len(X))
        return X[p], y[p]
    if cfg.get("with_lbr"):   # ANCHOR + SimLBR: the released-recipe blends are added to ANCHOR's negatives
        Zl = simlbr_blends()
        Xs.append(Zl); ys.append(np.ones(len(Zl)))
    if uses_fakes:
        if not len(tr_fake):
            raise SystemExit(f"{method} needs training fakes in the manifest")
        Xf = F[tr_fake].astype(np.float32)
        Xs.append(Xf); ys.append(np.ones(len(Xf)))
    if method == "sup":
        Z = None
    elif method == "lbr":
        n = len(tr_fake)
        ri = rng.integers(0, len(Xr), n); fi = rng.integers(0, len(tr_fake), n)
        al = rng.uniform(cfg["a_min"], cfg["a_max"], (n, 1)).astype(np.float32)
        Z = al * Xr[ri] + (1 - al) * F[tr_fake[fi]].astype(np.float32)
    else:
        if not np.any(RB):
            raise SystemExit("reconstruction features are empty: run `features` / `recon` first")
        n = len(Xr)
        ri = rng.permutation(n)
        kk = pick()                                           # which autoencoder each negative comes from
        H, own = Xr[ri], RB[kk, ri]
        if method == "recon":
            Z = own
        else:
            be = rng.uniform(cfg["b_min"], cfg["b_max"], (n, 1)).astype(np.float32)
            if method == "mix_aligned":
                d = own - H
            else:
                j = (ri + rng.integers(1, n, n)) % n          # a different real's reconstruction, same autoencoder
                d = RB[kk, j] - H
                if method == "mix_unpaired_matched":
                    scale = np.linalg.norm(own - H, axis=1, keepdims=True) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
                    d = d * scale
            Z = H + be * d                                    # = (1-b) h(x) + b h(partner)
    if Z is not None:
        Xs.append(Z); ys.append(np.ones(len(Z)))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    p = rng.permutation(len(X))
    return X[p], y[p]


# ----------------------------------------------------------------------------- 5. metrics (numpy, testable)
def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def conformal_tau(cal_scores, eps):
    n = len(cal_scores)
    k = int(math.ceil((n + 1) * (1 - eps)))
    return float("inf") if k > n else float(np.sort(cal_scores)[k - 1])


def split_idx(rows):
    lab = np.array([r["label"] for r in rows]); split = np.array([r["split"] for r in rows])
    src = np.array([r["source"] for r in rows])
    return lab, split, src


def evaluate(scores, deg_scores, rows, eps_list=(0.01, 0.05)):
    """scores are head LOGITS (no sigmoid saturation, so no ties at the conformal threshold);
    the SimLBR-style accuracy uses probability > 0.5, i.e. logit > 0."""
    lab, split, src = split_idx(rows)
    cal = scores[(split == "calib") & (lab == 0)]
    if len(cal) == 0:
        raise SystemExit("no calibration reals: rebuild the manifest with --calib_frac > 0 (e.g. 0.3)")
    for e in eps_list:
        if math.ceil((len(cal) + 1) * (1 - e)) > len(cal):
            print(f"warning: {len(cal)} calibration reals are too few for a {e:.0%} target (threshold = inf)")
    taus = {e: conformal_tau(cal, e) for e in eps_list}
    test_real = (split == "test") & (lab == 0)
    out = []
    for s in sorted(set(src[(split == "test") & (lab == 1)])):
        fake = (split == "test") & (src == s) & (lab == 1)
        own_real = (split == "test") & (src == s) & (lab == 0)
        rm = own_real if own_real.any() else test_real
        acc_f = float((scores[fake] > 0).mean()); acc_r = float((scores[rm] <= 0).mean())
        o = dict(source=str(s), n_fake=int(fake.sum()), acc=100 * (acc_f + acc_r) / 2,
                 fake_acc=100 * acc_f, real_acc=100 * acc_r,
                 auroc=100 * auroc(np.r_[np.zeros(rm.sum()), np.ones(fake.sum())], np.r_[scores[rm], scores[fake]]))
        for e, t in taus.items():
            o[f"tpr@{e:.0%}"] = 100 * float((scores[fake] > t).mean())
        out.append(o)
    summ = dict(n_calib=int(len(cal)))
    accs = np.array([o["acc"] for o in out])
    summ.update(mean_acc=float(accs.mean()), worst_acc=float(accs.min()),
                reliability=float((accs.mean() - 50) / accs.std()) if accs.std() > 0 else float("nan"))
    # generators whose fakes appear in training (A/B/+fakes see them) are "seen"; the rest are unseen
    seen = set(src[(split == "train") & (lab == 1)])
    unseen = np.array([o["source"] not in seen for o in out])
    for e, t in taus.items():
        tp = np.array([o[f"tpr@{e:.0%}"] for o in out])
        summ[f"mean_tpr@{e:.0%}"] = float(tp.mean()); summ[f"worst_tpr@{e:.0%}"] = float(tp.min())
        if unseen.any():
            summ[f"unseen_mean_tpr@{e:.0%}"] = float(tp[unseen].mean())
        summ[f"fpr_test_reals@{e:.0%}"] = 100 * float((scores[test_real] > t).mean())
        summ[f"fpr_degraded_reals@{e:.0%}"] = 100 * float((deg_scores[test_real] > t).mean())
    return out, summ


def paired_bootstrap(sa, sb, rows, eps=0.05, B=1000, seed=0):
    """CI for (method a - method b) in mean and worst-case TPR at matched false-alarm eps.
    sa, sb: one score array, or lists of arrays (one per training seed, same seed order).
    Each draw resamples calibration reals and each generator's test fakes ONCE; every seed of
    both methods sees that same resample, and the difference is averaged over seeds inside the
    draw, so the interval is for the seed-averaged difference (images are the resampling unit)."""
    sa = [sa] if isinstance(sa, np.ndarray) else list(sa)
    sb = [sb] if isinstance(sb, np.ndarray) else list(sb)
    assert len(sa) == len(sb) and len(sa) > 0
    lab, split, src = split_idx(rows)
    rng = np.random.default_rng(seed)
    cal_idx = np.where((split == "calib") & (lab == 0))[0]
    gens = sorted(set(src[(split == "test") & (lab == 1)]))
    fake_idx = {g: np.where((split == "test") & (src == g) & (lab == 1))[0] for g in gens}
    dm, dw = [], []
    for _ in range(B):
        c = rng.choice(cal_idx, len(cal_idx))
        fs = [rng.choice(fake_idx[g], len(fake_idx[g])) for g in gens]
        m_d, w_d = [], []
        for xa, xb in zip(sa, sb):
            ta, tb = conformal_tau(xa[c], eps), conformal_tau(xb[c], eps)
            ra = np.array([(xa[f] > ta).mean() for f in fs]) * 100
            rb = np.array([(xb[f] > tb).mean() for f in fs]) * 100
            m_d.append(ra.mean() - rb.mean()); w_d.append(ra.min() - rb.min())
        dm.append(np.mean(m_d)); dw.append(np.mean(w_d))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(mean=q(dm), worst=q(dw))


# ----------------------------------------------------------------------------- 4. train
def build_head(dim, dropout, linear=False):
    import torch.nn as nn
    if linear:   # linear probe on frozen features ("Simplicity Prevails" baseline)
        return nn.Sequential(nn.Linear(dim, 1))
    return nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))


def head_from_state(sd):
    """Rebuild a head (MLP or linear) from its state dict; dropout is irrelevant at eval."""
    linear = len(sd) == 2
    head = build_head(sd["0.weight"].shape[1], 0.0, linear=linear)
    head.load_state_dict(sd); return head.eval()


def fit_head(F, R, G, rows, method, cfg, a, seed, dev, tag="head", verbose=True):
    """Train one head with a fixed optimizer-step budget (a.steps; 0 = a.epochs)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    head = build_head(F.shape[1], a.dropout, getattr(a, "linear", False)).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=a.wd)
    lossf = nn.BCEWithLogitsLoss()
    # Budget: a fixed number of optimizer steps, the same for every condition (--steps).
    # Epoch-based training is badly under-trained on pilot-sized data: 10 epochs of 1-5k rows at
    # batch 500 is only 20-150 steps (SimLBR's 10 epochs on GenImage are ~3k steps), and conditions
    # with more rows (e.g. +deg) would silently get more updates. --steps 0 restores --epochs.
    step, ep = 0, 0
    while (step < a.steps) if a.steps > 0 else (ep < a.epochs):
        X, y = make_training_set(F, R, G, rows, method, rng, cfg)
        head.train(); tot, seen = 0.0, 0
        for s in range(0, len(X), a.batch):
            if a.steps > 0 and step >= a.steps:
                break
            xb = torch.from_numpy(X[s:s + a.batch]).to(dev); yb = torch.from_numpy(y[s:s + a.batch]).float().to(dev)
            loss = lossf(head(xb).squeeze(1), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(xb); seen += len(xb); step += 1
        ep += 1
        if verbose and (a.steps <= 0 or ep % 25 == 0 or step >= a.steps):
            print(f"[{tag} seed {seed}] epoch {ep} step {step}/{a.steps if a.steps > 0 else '-'} "
                  f"loss {tot / max(seen, 1):.4f}", flush=True)
    return head.eval()


def logits(head, M, dev):
    import torch
    with torch.no_grad():   # LOGITS: sigmoid saturates to exactly 1.0 in float32 and creates ties
        return head(torch.from_numpy(np.asarray(M, dtype=np.float32)).to(dev)).squeeze(1).cpu().numpy()


def train_cfg(a):
    return dict(a_min=a.a_min, a_max=a.a_max, b_min=a.b_min, b_max=a.b_max,
                degraded_reals=a.degraded_reals, with_fakes=a.with_fakes, with_lbr=a.with_lbr)


def cmd_train(a):
    import torch
    dev = torch.device("cpu") if a.device == "auto" else pick_device(a.device)  # heads are tiny
    W = Path(a.work)
    rows = read_manifest(W / "manifest.csv")
    F = np.load(W / "cls.npy", mmap_mode="r"); R = np.load(W / "cls_recon.npy", mmap_mode="r")
    G = np.load(W / "cls_degraded.npy", mmap_mode="r")
    cfg = train_cfg(a)
    bank_names = [b for b in a.bank.split(",") if b]
    if bank_names != ["sd"]:
        cfg["bank"] = load_bank(W, bank_names)
    tag0 = (a.method + ("+deg" if a.degraded_reals else "") + ("+fakes" if a.with_fakes else "")
            + ("+lbr" if a.with_lbr else "") + ("+lin" if a.linear else "") + (f"~{a.tag}" if a.tag else "")
            + ("" if bank_names == ["sd"] else "@" + "-".join(bank_names)))
    done_path = W / "features_done.json"
    n_done = len(json.load(open(done_path))) if done_path.exists() else 0
    if n_done < len(rows):
        raise SystemExit(f"features cached for {n_done}/{len(rows)} rows: finish `features` first")
    for seed in a.seeds:
        head = fit_head(F, R, G, rows, a.method, cfg, a, seed, dev, tag=tag0)
        np.save(W / f"scores_{tag0}_seed{seed}.npy", logits(head, F, dev))
        np.save(W / f"degscores_{tag0}_seed{seed}.npy", logits(head, G, dev))
        torch.save(head.state_dict(), W / f"head_{tag0}_seed{seed}.pt")
        print(f"saved scores_{tag0}_seed{seed}.npy")


# ----------------------------------------------------------------------------- 4a. beta selection (zero-fake, Z2)
def cmd_select(a):
    """Choose ANCHOR's beta range WITHOUT any generated image (zero-fake level Z2).
    For each held-out autoencoder family k: fit on 80% of training reals with the bank minus k, then
    measure how well the head flags family k's reconstructions of the other 20% of reals (TPR at the
    95th percentile of those reals' scores). The range with the best mean over k is chosen."""
    import torch
    dev = torch.device("cpu")
    W = Path(a.work)
    rows = read_manifest(W / "manifest.csv")
    F = np.load(W / "cls.npy", mmap_mode="r"); G = np.load(W / "cls_degraded.npy", mmap_mode="r")
    names = [b for b in a.bank.split(",") if b]
    bank = dict(zip(names, load_bank(W, names)))
    tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0])
    rng = np.random.default_rng(a.seed)
    val = set(rng.choice(tr, len(tr) // 5, replace=False).tolist())
    fit_rows = [dict(r, split="val") if i in val else r for i, r in enumerate(rows)]
    vi = np.array(sorted(val))
    ranges = [tuple(float(x) for x in r.split("-")) for r in a.ranges.split(",")]
    res = {}
    for m in a.methods.split(","):
        for (lo, hi) in (ranges if m != "recon" else [(1.0, 1.0)]):
            per_k = {}
            for k in names:
                cfg = dict(a_min=0.5, a_max=0.8, b_min=lo, b_max=hi, degraded_reals=False, with_fakes=False,
                           with_lbr=False, bank=[bank[n] for n in names if n != k])
                head = fit_head(F, None, G, fit_rows, m, cfg, a, 0, dev, verbose=False)
                sr = logits(head, F[vi], dev); sk = logits(head, bank[k][0][vi], dev)
                per_k[k] = 100 * float((sk > np.percentile(sr, 95)).mean())
            key = f"{m}~b{lo:g}-{hi:g}"
            res[key] = dict(method=m, b_min=lo, b_max=hi, per_family=per_k, mean=float(np.mean(list(per_k.values()))))
            print(f"{key:34s} held-out-family TPR@5%: " + "  ".join(f"{k} {v:5.1f}" for k, v in per_k.items())
                  + f"   mean {res[key]['mean']:5.1f}", flush=True)
    chosen = {}
    for m in a.methods.split(","):
        if m == "recon":
            continue
        best = max((v for v in res.values() if v["method"] == m), key=lambda v: v["mean"])
        chosen[m] = [best["b_min"], best["b_max"]]
        print(f"chosen for {m}: beta ~ U({best['b_min']:g}, {best['b_max']:g})  (no generated image used)")
    json.dump(dict(results=res, chosen=chosen), open(W / "select_beta.json", "w"), indent=1)


# ----------------------------------------------------------------------------- 4b. score (transfer)
def cmd_score(a):
    """Score heads trained in another work folder (--heads_from) on THIS folder's cached features,
    e.g. a test-only Chameleon folder. Writes scores_/degscores_ files, so `eval` works unchanged.
    Both folders must use the same backbone and image size (checked against features_meta.json)."""
    import torch, torch.nn as nn
    W = Path(a.work)
    rows = read_manifest(W / "manifest.csv")
    if a.official:
        return score_official(a, W, rows)
    S = Path(a.heads_from)
    mw, ms = json.load(open(W / "features_meta.json")), json.load(open(S / "features_meta.json"))
    mw.setdefault("preproc", "crop"); ms.setdefault("preproc", "crop")
    for k in ("backbone", "size", "dim", "preproc"):
        if mw.get(k) != ms.get(k):
            raise SystemExit(f"{k} differs: {W} has {mw.get(k)}, {S} has {ms.get(k)}")
    done_path = W / "features_done.json"
    if len(json.load(open(done_path))) < len(rows):
        raise SystemExit(f"features in {W} are not finished")
    F = np.asarray(np.load(W / "cls.npy", mmap_mode="r"), dtype=np.float32)
    G = np.asarray(np.load(W / "cls_degraded.npy", mmap_mode="r"), dtype=np.float32)
    heads = sorted(S.glob("head_*_seed*.pt"))
    if not heads:
        raise SystemExit(f"no head_*.pt in {S}")
    n_new = 0
    for h in heads:
        tag = h.stem[len("head_"):]
        out = W / f"scores_{tag}.npy"
        if out.exists() and out.stat().st_mtime >= h.stat().st_mtime:
            continue
        head = head_from_state(torch.load(h, map_location="cpu"))
        with torch.no_grad():
            np.save(out, head(torch.from_numpy(F)).squeeze(1).numpy())
            np.save(W / f"degscores_{tag}.npy", head(torch.from_numpy(G)).squeeze(1).numpy())
        n_new += 1
    print(f"scored {n_new} new heads from {S} on {len(rows)} rows of {W} ({len(heads)} heads in total)")


def score_official(a, W, rows):
    """Released SimLBR head (MVRL/SimLBR, variant a.official) on this folder's features, which must be
    made with `features --preproc simlbr`. Scores are written to --out (default: this folder), whose
    manifest must list the same rows, as scores_simlbr_official_seed{0,1,2} (one model, copied so
    paired comparisons with 3-seed methods use every seed of those methods)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    meta = json.load(open(W / "features_meta.json"))
    if meta.get("preproc") != "simlbr" or "vitl16" not in meta["backbone"]:
        raise SystemExit("the released SimLBR heads need DINOv3 ViT-L/16 features made with --preproc simlbr")
    out = Path(a.out or W)
    if [r["path"] for r in read_manifest(out / "manifest.csv")] != [r["path"] for r in rows]:
        raise SystemExit(f"{out} lists different rows than {W}")
    head = head_from_state(load_file(hf_hub_download("MVRL/SimLBR", f"{a.official}/model.safetensors")))
    F = np.load(W / "cls.npy", mmap_mode="r"); G = np.load(W / "cls_degraded.npy", mmap_mode="r")
    s, g = logits(head, F, torch.device("cpu")), logits(head, G, torch.device("cpu"))
    for seed in (0, 1, 2):
        np.save(out / f"scores_simlbr_official_seed{seed}.npy", s)
        np.save(out / f"degscores_simlbr_official_seed{seed}.npy", g)
    lab = np.array([r["label"] for r in rows]); sp = np.array([r["split"] for r in rows])
    m = sp != "train"
    print(f"released SimLBR ({a.official}) accuracy at 0.5 on {m.sum()} eval rows: "
          f"{100 * ((s[m] > 0) == (lab[m] == 1)).mean():.1f}%  -> wrote {out}/scores_simlbr_official_seed*.npy")


# ----------------------------------------------------------------------------- 5. eval
def cmd_eval(a):
    import pandas as pd
    W = Path(a.work)
    rows = read_manifest(W / "manifest.csv")
    files = sorted(W.glob("scores_*_seed*.npy"))
    if not files:
        raise SystemExit("no scores found: run `train` first")
    runs = {}
    for f in files:
        tag = f.stem[len("scores_"):]
        method, seed = tag.rsplit("_seed", 1)
        runs.setdefault(method, {})[int(seed)] = (np.load(f), np.load(W / f"degscores_{tag}.npy"))
    per_gen, summ = [], []
    for m, bys in runs.items():
        for seed, (s, g) in bys.items():
            out, sm = evaluate(s, g, rows)
            for o in out:
                o.update(method=m, seed=seed); per_gen.append(o)
            sm.update(method=m, seed=seed); summ.append(sm)
    pg, sm = pd.DataFrame(per_gen), pd.DataFrame(summ)
    pg.to_csv(W / "results_per_generator.csv", index=False); sm.to_csv(W / "results_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPer generator, mean over seeds (TPR = fake-detection rate at the conformal threshold):")
    print(pg.groupby(["method", "source"])[["auroc", "acc", "tpr@1%", "tpr@5%"]].mean().round(1).to_string())
    cols = ["mean_tpr@5%", "worst_tpr@5%"] + (["unseen_mean_tpr@5%"] if "unseen_mean_tpr@5%" in sm else []) + ["mean_tpr@1%", "worst_tpr@1%", "fpr_test_reals@5%", "fpr_degraded_reals@5%", "mean_acc", "reliability"]
    print("\nSummary, mean over seeds (all thresholds set on the SAME calibration reals):")
    t = sm.groupby("method")[cols].mean().round(1)
    t.insert(0, "cond", [LETTER.get(base_method(m), "?") for m in t.index])
    print(t.to_string())
    # paired comparisons for the decisive question
    base = base_method
    E = sorted(m for m in runs if base(m) == "mix_aligned")
    cis, report = {}, {}
    for e in E:
        suffix = e[len("mix_aligned"):]
        for other in ("recon", "mix_unpaired_matched", "mix_unpaired") + SIMLBR_TAGS:
            o = other + suffix if other not in SIMLBR_TAGS else other
            if o not in runs:
                continue
            seeds = sorted(set(runs[e]) & set(runs[o]))
            if not seeds:
                print(f"\n{e} vs {o}: no common seeds, skipped"); continue
            r = paired_bootstrap([runs[e][sd][0] for sd in seeds], [runs[o][sd][0] for sd in seeds],
                                 rows, eps=0.05, B=a.boot, seed=a.seed)
            cis[(e, o)] = r
            (mm, ml, mh), (wm, wl, wh) = r["mean"], r["worst"]
            print(f"\nE({e}) - {LETTER.get(other)}({o}), TPR at 5% false alarms, averaged over {len(seeds)} seeds:"
                  f"\n   mean  {mm:+.1f} pts  [95% CI {ml:+.1f}, {mh:+.1f}]"
                  f"\n   worst {wm:+.1f} pts  [95% CI {wl:+.1f}, {wh:+.1f}]")
    # go / no-go, for every E variant (plain and +deg)
    g, w, dg = t["mean_tpr@5%"], t["worst_tpr@5%"], t["fpr_degraded_reals@5%"]
    for e in E:
        sfx = e[len("mix_aligned"):]
        c, d = "recon" + sfx, "mix_unpaired_matched" + sfx
        if not all(m in t.index for m in (c, d)):
            print(f"\nGO / NO-GO for {e}: needs {c} and {d} trained first"); continue
        beats = bool(w[e] - w[c] >= 2 and w[e] - w[d] >= 2 and g[e] - g[c] >= -1 and g[e] - g[d] >= -1)
        clean = bool(dg[e] <= dg[c] + 1)
        ci_ok = all((e, o) in cis and cis[(e, o)]["worst"][1] > 0 for o in (c, d))
        verdict = "GO (finding)" if beats and clean and ci_ok else ("GO (point estimate only; CI includes 0)"
                                                                   if beats and clean else "NO-GO on H1")
        report[e] = dict(beats_C_and_Dstar=beats, degraded_ok=clean, worst_ci_excludes_0=ci_ok, verdict=verdict)
        print(f"\nGO / NO-GO for {e} (suggested rule):"
              f"\n   worst-case TPR@5%: E-C {w[e] - w[c]:+.1f}, E-D* {w[e] - w[d]:+.1f} (need >= +2 both)"
              f"\n   mean TPR@5%:       E-C {g[e] - g[c]:+.1f}, E-D* {g[e] - g[d]:+.1f} (need >= -1 both)"
              f"\n   false alarms on degraded reals: E {dg[e]:.1f}% vs C {dg[c]:.1f}% (need E <= C + 1)"
              f"\n   paired 95% CIs of worst-case gain exclude 0: {ci_ok}"
              f"\n   ->  {verdict}")
    # head-to-head against SimLBR (B) for every E variant; the +fakes variants are the fair comparison
    h2h = {}
    for bt in SIMLBR_TAGS:
      if E and any((e, bt) in cis for e in E):
        print(f"\nHEAD-TO-HEAD vs SimLBR ({bt}), TPR at 5% false alarms (paired 95% CI):")
        for e in E:
            if (e, bt) in cis:
                r = cis[(e, bt)]
                win = r["mean"][1] > 0 and r["worst"][1] > 0
                tie = r["mean"][1] <= 0 <= r["mean"][2]
                key = e if bt == "lbr" else f"{e} vs {bt}"
                h2h[key] = dict(mean=r["mean"], worst=r["worst"],
                                verdict="beats SimLBR" if win else ("ties SimLBR" if tie else "below SimLBR"))
                print(f"   {e:34s} mean {r['mean'][0]:+5.1f} [{r['mean'][1]:+.1f}, {r['mean'][2]:+.1f}]   "
                      f"worst {r['worst'][0]:+5.1f} [{r['worst'][1]:+.1f}, {r['worst'][2]:+.1f}]  -> {h2h[key]['verdict']}")
    json.dump(dict(go_no_go=report, vs_simlbr=h2h, finished=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(W / "eval_done.json", "w"), indent=1)
    print(f"\nwrote {W / 'results_summary.csv'}, {W / 'results_per_generator.csv'}, {W / 'eval_done.json'}")


# ----------------------------------------------------------------------------- 6. status
def cmd_status(a):
    """How far the pilot has got: manifest -> features -> heads -> eval."""
    W = Path(a.work)
    plan = W / "planned_heads.txt"   # written by run_pilot.sh: one head tag per line
    expected = ([l.strip() for l in open(plan) if l.strip()] if plan.exists() else
                [m for m in METHODS] + [m + "+deg" for m in ("recon", "mix_unpaired_matched", "mix_aligned")])
    man = W / "manifest.csv"
    if not man.exists():
        print(f"[ ] 1 manifest   not built in {W}  (run `manifest`)"); return
    rows = read_manifest(man); N = len(rows)
    print(f"[x] 1 manifest   {N} images")
    done_path = W / "features_done.json"
    nd = len(json.load(open(done_path))) if done_path.exists() else 0
    bar = "#" * int(30 * nd / max(N, 1))
    age = f", last checkpoint {time.time() - done_path.stat().st_mtime:.0f}s ago" if done_path.exists() else ""
    print(f"[{'x' if nd == N else ' '}] 2 features   {nd}/{N} ({100 * nd / max(N, 1):.1f}%) |{bar:<30}|{age}")
    have = {}
    for f in W.glob("scores_*_seed*.npy"):
        m, sd = f.stem[len("scores_"):].rsplit("_seed", 1); have.setdefault(m, set()).add(int(sd))
    n_heads = sum(len(v) for v in have.values())
    print(f"[{'x' if all(m in have for m in expected) else ' '}] 3 train      {n_heads} heads saved")
    for m in expected + sorted(set(have) - set(expected)):
        s = sorted(have.get(m, []))
        print(f"       {LETTER.get(base_method(m), '?'):2s} {m:28s} seeds {s if s else '-'}")
    ev = W / "eval_done.json"
    newest = max((f.stat().st_mtime for f in W.glob("scores_*_seed*.npy")), default=0)
    if ev.exists() and ev.stat().st_mtime < newest:
        print("[ ] 4 eval       out of date: heads were retrained after the last eval (re-run `eval`)")
    elif ev.exists():
        info = json.load(open(ev))
        print(f"[x] 4 eval       finished {info.get('finished')}")
        for e, r in info.get("go_no_go", {}).items():
            print(f"       {e}: {r['verdict']}")
        for e, r in info.get("vs_simlbr", {}).items():
            print(f"       {e} vs SimLBR: {r['verdict']}")
    else:
        print("[ ] 4 eval       not run yet")


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="anchor_work"); p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--entry", action="append", required=True)
    m.add_argument("--calib_frac", type=float, default=0.3, help="share of test reals moved to the calib split")
    b = sub.add_parser("bench"); b.add_argument("--size", type=int, default=256); b.add_argument("--batch", type=int, default=16)
    f = sub.add_parser("features"); f.add_argument("--size", type=int, default=256); f.add_argument("--batch", type=int, default=16)
    f.add_argument("--jpeg_prob", type=float, default=0.5); f.add_argument("--jpeg_min", type=int, default=70)
    f.add_argument("--test_jpeg", type=int, default=0,
                   help="re-save EVERY calib/test image (real and fake) as JPEG at this quality (0 = off); "
                        "removes the GenImage real=JPEG / fake=PNG format shortcut at test time")
    f.add_argument("--preproc", choices=["crop", "simlbr"], default="crop",
                   help="simlbr = resize to 256x256 and SimLBR's normalisation (needed for the released SimLBR heads)")
    f.add_argument("--skip_train", action="store_true", help="cache calib/test rows only (evaluation copies)")
    for s in (b, f):
        s.add_argument("--backbone", default=DINO_ID, help="Hugging Face id of the frozen backbone")
        s.add_argument("--vae", default=VAE_ID, help="autoencoder id; REPO:SUBFOLDER for a pipeline's vae")
        s.add_argument("--vae_batch", type=int, default=4, help="VAE sub-batch (keep 4-8 on a 16 GB Mac)")
    t = sub.add_parser("train"); t.add_argument("--method", choices=METHODS, required=True)
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--batch", type=int, default=500)
    t.add_argument("--steps", type=int, default=2000, help="optimizer steps per head, same for every condition (0 = use --epochs)")
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--dropout", type=float, default=0.5)
    t.add_argument("--a_min", type=float, default=0.5); t.add_argument("--a_max", type=float, default=0.8)
    t.add_argument("--b_min", type=float, default=0.5); t.add_argument("--b_max", type=float, default=2.0)
    t.add_argument("--degraded_reals", action="store_true", help="add degradation-matched reals to training, labelled real")
    t.add_argument("--with_fakes", action="store_true", help="also add actual training fakes (breaks the zero-fake claim)")
    t.add_argument("--bank", default="sd", help="reconstruction bank, comma-separated `recon` names; 'sd' = the `features` VAE")
    t.add_argument("--with_lbr", action="store_true", help="ANCHOR + SimLBR: add released-recipe SimLBR blends as negatives")
    t.add_argument("--linear", action="store_true", help="linear probe instead of the MLP head")
    t.add_argument("--tag", default="", help="extra tag suffix, e.g. b0.5-1.0 for a beta sweep")
    sl = sub.add_parser("select", help="choose beta on held-out autoencoder families (no generated images)")
    sl.add_argument("--bank", default="sd,flux,vq,dcae"); sl.add_argument("--methods", default="recon,mix_aligned,mix_unpaired_matched")
    sl.add_argument("--ranges", default="0.25-0.75,0.5-1,0.5-2,1-2,1-3")
    sl.add_argument("--steps", type=int, default=2000); sl.add_argument("--epochs", type=int, default=10)
    sl.add_argument("--batch", type=int, default=500); sl.add_argument("--lr", type=float, default=1e-4)
    sl.add_argument("--wd", type=float, default=1e-2); sl.add_argument("--dropout", type=float, default=0.5)
    r = sub.add_parser("recon", help="add one autoencoder to the reconstruction bank")
    r.add_argument("--vae", required=True); r.add_argument("--name", required=True)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--vae_batch", type=int, default=4)
    e = sub.add_parser("eval"); e.add_argument("--boot", type=int, default=1000)
    sc = sub.add_parser("score", help="score heads trained in another work folder on this folder's features")
    sc.add_argument("--heads_from", default=None)
    sc.add_argument("--official", choices=["genimage", "aigc"], default=None, help="score the released SimLBR head instead")
    sc.add_argument("--out", default=None, help="with --official: folder (same manifest) to write the scores into")
    sub.add_parser("status")
    a = p.parse_args(argv)
    {"manifest": cmd_manifest, "bench": cmd_bench, "features": cmd_features, "train": cmd_train,
     "eval": cmd_eval, "status": cmd_status, "recon": cmd_recon, "score": cmd_score,
     "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
