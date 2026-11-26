"""Déjà VAE pilot: generated images have already been through a VAE decoder, so a second trip
through the autoencoder barely moves them. Real photos move more.

Pair features (frozen DINOv3 CLS) for every image view v:   a = phi(v),  b = phi(R(v))
where R = SD-VAE encode->decode applied at NATIVE resolution (centre crop <= 256 px, aligned to
the 8-px latent grid, never resized before R; the pilot showed that resizing first destroys the
fixed point). Only the DINOv3 input is resized to 256 when a view is smaller.

Zero-fake training pairs, from training reals x only (no generated image anywhere):
   real            (J(x),   R(J(x)))
   simulated fake  (J(R(x)), R(J(R(x))))        a reconstruction IS a decoder output
   counterfactual  (J(x~),  R(J(x~)))           x~ = blur with PSNR(x~, x) = PSNR(R(x), x), labelled real
J = JPEG re-save at q ~ U(70,100) per image (test images are all re-saved at q=95, the p2 protocol).

Commands (all resumable):
   pairs  --set train|genimage|cham      cache pair features
   train                                  heads: dv, dv-nocf, dv-single (+ training-free dv-tf)
   report --set genimage|cham             scores + paired bootstrap vs SimLBR / ANCHOR heads
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap

W = Path(os.environ.get("DV_WORK", "runs/dv"))
SETS = {   # name -> (manifest, folder holding the comparison methods' scores, test JPEG quality)
    "train": ("runs/p2/manifest.csv", None, None),
    "genimage": ("runs/p2/manifest.csv", "runs/p4s", 95),
    "cham": ("runs/cham/manifest.csv", "runs/cham_off", None),
    "cham_m": ("runs/cham_m/manifest.csv", "runs/cham_m", None),   # E6: fakes size-matched to the reals (cham_match.py)
    "cham_c": ("runs/cham_c/manifest.csv", "runs/cham_c", None),   # E6 control: same resize + JPEG, own aspect, no crop
    "sb": ("runs/sb/manifest.csv", "runs/sb", 95),                 # E5: Synthbuster + geometry-matched RAISE-1k reals
    "sb_crop": ("runs/sb_crop/manifest.csv", "runs/sb_crop", 95),  # review: reals = native-resolution crops (no resampling)
    "sb_raw": ("runs/sb/manifest.csv", "runs/sb_raw", None),       # review: Synthbuster without the JPEG q95 re-save
    "eg": ("runs/eg/manifest.csv", "runs/eg", 95),                 # revision: EvalGEN + RAISE reals downscaled to 1024
    "eg_crop": ("runs/eg_crop/manifest.csv", "runs/eg_crop", 95),  # revision: EvalGEN + RAISE reals cropped to 1024
    "e5": ("runs/e5/manifest.csv", "runs/e5", 95),                 # revision E5: VISION phone photos (real only)
    "gi_c": ("runs/gi_c/manifest.csv", "runs/gi_c", 95),           # revision: GenImage subset, processing-only control
    "gi_m": ("runs/gi_m/manifest.csv", "runs/gi_m", 95),           # revision: GenImage subset, size-matched
}
N_TRAIN = int(os.environ.get("DV_NTRAIN", 5000))
LIMIT = int(os.environ.get("DV_LIMIT", 0))   # smoke test: keep ~LIMIT test rows per (split, label, source)
FEAT = os.environ.get("DV_FEAT", "cls24")   # cls24: final-layer CLS (pilot v1); mid: CLS of middle layers (v2)
MID = [8, 10, 12, 14, 16, 18]               # chosen from the training-free layer diagnostic (dv_layers.py)
D = 1024 * (len(MID) if FEAT == "mid" else 1)
AUG = os.environ.get("DV_AUG", "")            # "post" (v3): in-the-wild processing on every training row
S_GRID = [0.5, 0.625, 0.75, 0.875, 1.25, 1.5]  # resize factors (quantised so the VAE still batches by size)
AE = os.environ.get("DV_AE", ap.VAE_ID)       # autoencoder that makes the copy R (default: SD ft-MSE, as v1-v3)
TF_LAYER = 16                               # training-free score layer in v2 (best mean AUROC in the diagnostic)
COMPARE = ["lbr_exact", "simlbr_official", "mix_aligned+lbr", "recon+lbr", "recon", "sup", "rigid-dinov2", "rigid-dinov3", "rigid-dinov2~norm"]


def view(path):
    """Native-resolution centre crop, at most 256 px a side, offsets and size on the 8-px grid."""
    from PIL import Image
    im = Image.open(path).convert("RGB"); w, h = im.size
    cw, ch = min(256, w // 8 * 8), min(256, h // 8 * 8)
    l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return im.crop((l, t, l + cw, t + ch))


def to256(im):
    from PIL import Image
    return im if im.size == (256, 256) else im.resize((256, 256), Image.BICUBIC)


class Pairs:
    """Groups images by size so the VAE runs batched; returns (phi(v), phi(R(v)))."""
    def __init__(self, enc, vae_batch=4):
        self.enc, self.vb = enc, vae_batch

    def recon(self, ims):
        out = [None] * len(ims); by = {}
        for i, im in enumerate(ims):
            by.setdefault(im.size, []).append(i)
        for _, idx in by.items():
            for i, r in zip(idx, self.enc.reconstruct([ims[i] for i in idx], self.vb)):
                out[i] = r
        return out

    def phi(self, ims):
        if FEAT != "mid":
            return self.enc.cls([to256(im) for im in ims]).astype(np.float16)
        e = self.enc
        with e.torch.no_grad():
            x = (e.to_tensor([to256(im) for im in ims]) - e.mean) / e.std
            hs = e.dino(pixel_values=x, output_hidden_states=True).hidden_states
            return e.torch.cat([hs[L][:, 0] for L in MID], 1).float().cpu().numpy().astype(np.float16)


def rows_of(name):
    man, _, _ = SETS[name]
    rows = ap.read_manifest(Path(man))
    if name == "train":
        tr = [i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == 0]
        rng = np.random.default_rng(0)
        return rows, sorted(rng.choice(tr, N_TRAIN, replace=False).tolist())
    idx = [i for i, r in enumerate(rows) if r["split"] != "train"]
    if LIMIT:   # smoke test: keep the first LIMIT rows of each group, other rows stay out of the eval
        seen, keep = {}, []
        for i in idx:
            g = (rows[i]["split"], rows[i]["label"], rows[i]["source"]); seen[g] = seen.get(g, 0) + 1
            if seen[g] <= LIMIT: keep.append(i)
        idx = keep
    return rows, idx


def cmd_pairs(a):
    from tqdm import tqdm
    from PIL import ImageFilter
    W.mkdir(parents=True, exist_ok=True)
    rows, idx = rows_of(a.set)
    N = len(rows)
    kinds = ["real", "fake", "cf"] if a.set == "train" else ["img", "deg"]
    mm = {k: {s: np.lib.format.open_memmap(W / f"{a.set}_{k}_{s}.npy", mode="r+" if (W / f"{a.set}_{k}_{s}.npy").exists() else "w+",
                                           dtype=np.float16, shape=(N, D)) for s in "ab"} for k in kinds}
    done_p = W / f"{a.set}_done.json"
    done = set(json.load(open(done_p))) if done_p.exists() else set()
    rng = np.random.default_rng(0)                      # same per-row draws as `features` (seed 0)
    q = np.where(rng.random(N) < 1.0, rng.integers(70, 101, N), 0)
    sig = rng.uniform(0.5, 1.5, N)
    tq = SETS[a.set][2]
    arng = np.random.default_rng(1)                     # v3 processing draws (separate stream: q/sig stay as in v2)
    a_s = np.where(arng.random(N) < 0.3, 1.0, np.array(S_GRID)[arng.integers(0, len(S_GRID), N)])
    a_q = np.where(arng.random(N) < 0.5, arng.integers(60, 91, N), 0)
    def proc(im, i):
        """generate -> (upload JPEG) -> resize -> our native-res crop; identical for a row's real / fake / cf."""
        if a_q[i]:
            im = ap.jpeg(im, a_q[i])
        if a_s[i] != 1.0:
            from PIL import Image
            im = im.resize((max(8, round(im.width * a_s[i])), max(8, round(im.height * a_s[i]))), Image.BICUBIC)
        w, h = im.size; cw, ch = min(256, w // 8 * 8), min(256, h // 8 * 8)
        l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
        return im.crop((l, t, l + cw, t + ch))
    P_ = (lambda im, i: proc(im, i)) if (AUG == "post" and a.set == "train") else (lambda im, i: im)
    J = (lambda im, i: ap.jpeg(im, q[i])) if a.set == "train" else ((lambda im, i: ap.jpeg(im, tq)) if tq else (lambda im, i: im))
    tag = W / "autoencoder.txt"        # one autoencoder per work folder: never mix copies made with different decoders
    if tag.exists() and tag.read_text().strip() != AE:
        raise SystemExit(f"{W} was built with {tag.read_text().strip()}, not {AE}")
    W.mkdir(parents=True, exist_ok=True); tag.write_text(AE + "\n")
    enc = ap.Encoders(ap.pick_device(), norm="simlbr", vae=AE)
    P = Pairs(enc)
    todo = [i for i in idx if i not in done]
    print(f"{a.set}: {len(todo)} of {len(idx)} rows to do", flush=True)
    t0 = time.time()
    for s in tqdm(range(0, len(todo), a.batch), desc=f"pairs {a.set}"):
        b = todo[s:s + a.batch]
        x = [view(rows[i]["path"]) for i in b]
        A = {}
        if a.set == "train":
            R0 = P.recon(x)
            xt = [ap.blur_to_psnr(v, ap.psnr(v, r)) for v, r in zip(x, R0)]
            A["real"] = [J(P_(v, i), i) for v, i in zip(x, b)]
            A["fake"] = [J(P_(r, i), i) for r, i in zip(R0, b)]
            A["cf"] = [J(P_(v, i), i) for v, i in zip(xt, b)]
        else:
            A["img"] = [J(v, i) for v, i in zip(x, b)]
            reals = [k for k, i in enumerate(b) if rows[i]["label"] == 0 and rows[i]["split"] == "test"]
            A["deg"] = [J(x[k].filter(ImageFilter.GaussianBlur(float(sig[b[k]]))), b[k]) for k in reals]
        for k, ims in A.items():
            if not ims:
                continue
            pos = b if k != "deg" else [b[j] for j in reals]
            mm[k]["a"][pos] = P.phi(ims)
            mm[k]["b"][pos] = P.phi(P.recon(ims))
        done.update(b)
        if (s // a.batch) % 20 == 0:
            for k in mm: mm[k]["a"].flush(); mm[k]["b"].flush()
            json.dump(sorted(done), open(done_p, "w"))
        if s == 0:
            print(f"first batch {len(b) / (time.time() - t0):.2f} rows/s", flush=True)
    for k in mm: mm[k]["a"].flush(); mm[k]["b"].flush()
    json.dump(sorted(done), open(done_p, "w"))
    print(f"PAIRS {a.set} DONE: {len(done)} rows", flush=True)


def rel(a, b, mode):
    a = a.astype(np.float32); b = b.astype(np.float32)
    if mode == "single":
        return a
    if mode == "prof":   # per-layer distance profile: cosine distance, relative L2 change, both norms (log)
        n = a.shape[1] // 1024
        A, B = a.reshape(len(a), n, 1024), b.reshape(len(b), n, 1024)
        na, nb = np.linalg.norm(A, axis=2), np.linalg.norm(B, axis=2)
        cos = (A * B).sum(2) / (na * nb + 1e-8)
        f = [np.log(np.clip(1 - cos, 1e-6, None)), np.log(np.linalg.norm(A - B, axis=2) / (na + 1e-8) + 1e-6), np.log(na + 1e-6), np.log(nb + 1e-6)]
        return np.concatenate(f, 1)
    return np.concatenate([a, a - b, a * b], 1)


def rel_chunked(a, b, mode, fn=None, chunk=4096):
    """rel() over row chunks (mid-layer rel features are 18k-dim: 52k rows would need ~4 GB at once)."""
    out = []
    for s in range(0, len(a), chunk):
        r = rel(np.asarray(a[s:s + chunk]), np.asarray(b[s:s + chunk]), mode)
        out.append(fn(r) if fn else r.astype(np.float16))
    return np.concatenate(out)


VARIANTS = {"dv": ("rel", True), "dv-nocf": ("rel", False), "dv-single": ("single", True),
            "dv-single-nocf": ("single", False)}   # review ablation: image-only head without counterfactual rows
if FEAT == "mid":
    VARIANTS["dv-prof"] = ("prof", True)


def cmd_train(a):
    import torch, torch.nn as nn
    rows, idx = rows_of("train")
    ld = lambda k, s: np.load(W / f"train_{k}_{s}.npy", mmap_mode="r")[idx]
    P = {k: (ld(k, "a"), ld(k, "b")) for k in ("real", "fake", "cf")}
    for name, (mode, cf) in VARIANTS.items():
        parts = [(P["real"], 0), (P["fake"], 1)] + ([(P["cf"], 0)] if cf else [])
        X = np.concatenate([rel_chunked(pa, pb, mode) for (pa, pb), _ in parts])   # float16; batches cast below
        y = np.concatenate([np.full(len(pa), lab, np.float32) for (pa, _), lab in parts])
        pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])     # balance real : fake
        for seed in a.seeds:
            out = W / f"head_{name}_seed{seed}.pt"
            if out.exists():
                continue
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            head = ap.build_head(X.shape[1], a.dropout)
            opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=a.wd)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
            head.train()
            for step in range(a.steps):
                bi = rng.integers(0, len(X), a.batch)
                xb = torch.from_numpy(X[bi].astype(np.float32)); yb = torch.from_numpy(y[bi])
                loss = lossf(head(xb).squeeze(1), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            torch.save(head.state_dict(), out)
            print(f"trained {name} seed {seed}  final loss {loss.item():.3f}", flush=True)


def scores_for(name, rows, idx=None):
    """(scores, degscores) per seed for one Déjà VAE variant on a test set."""
    import torch
    S = {}
    N = len(rows)
    for k in ("img", "deg"):
        S[k] = (np.load(W / f"{name}_{k}_a.npy", mmap_mode="r"), np.load(W / f"{name}_{k}_b.npy", mmap_mode="r"))
    ix = np.arange(N) if idx is None else np.asarray(idx)   # only rows with features; others stay 0 and are cut later
    out = {}
    for v, (mode, _) in VARIANTS.items():
        if not (W / f"head_{v}_seed0.pt").exists():   # variant not trained in this work folder
            continue
        for seed in (0, 1, 2):
            head = ap.head_from_state(torch.load(W / f"head_{v}_seed{seed}.pt", map_location="cpu"))
            def f(k):
                sc = np.zeros(N, np.float32)
                with torch.no_grad():
                    sc[ix] = rel_chunked(S[k][0][ix], S[k][1][ix], mode,
                                         fn=lambda r: head(torch.from_numpy(r)).squeeze(1).numpy())
                return sc
            out.setdefault(v, {})[seed] = (f("img"), f("deg"))
    def tf(pa, pb):   # training-free: closer to a fixed point = more likely generated
        pa = pa.astype(np.float32); pb = pb.astype(np.float32)
        c = (pa * pb).sum(1) / (np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + 1e-8)
        return -np.log(np.clip(1 - c, 1e-6, None))
    if FEAT == "mid":   # training-free score on one middle layer
        j = MID.index(TF_LAYER)
        sl = lambda k: tuple(x[:, j * 1024:(j + 1) * 1024] for x in S[k])
        t = (tf(*sl("img")), tf(*sl("deg")))
    else:
        t = (tf(*S["img"]), tf(*S["deg"]))
    out["dv-tf"] = {s: t for s in (0, 1, 2)}
    return out


def zfuse(s1, s2, rows):
    lab, split, _ = ap.split_idx(rows)
    cal = (split == "calib") & (lab == 0)
    z = lambda s, ref: (s - ref[cal].mean()) / (ref[cal].std() + 1e-8)
    return lambda x1, x2: z(x1, s1) + z(x2, s2)


def cmd_report(a):
    import pandas as pd
    rows, idx = rows_of(a.set)
    ref = Path(SETS[a.set][1])
    runs = scores_for(a.set, rows, idx)
    for m in COMPARE:
        if (ref / f"scores_{m}_seed0.npy").exists():
            runs[m] = {s: (np.load(ref / f"scores_{m}_seed{s}.npy"), np.load(ref / f"degscores_{m}_seed{s}.npy")) for s in (0, 1, 2)}
    ix = np.array(idx)   # evaluate only the rows whose pair features exist (all calib/test rows in the full run)
    runs = {m: {s: (sc[ix], dg[ix]) for s, (sc, dg) in bys.items()} for m, bys in runs.items()}
    rows = [rows[i] for i in idx]
    for v in ("dv", "dv-tf", "dv-single", "dv-nocf", "dv-prof", "recon", "sup", "rigid-dinov2"):   # score fusion with SimLBR, z-scored on calib reals
        for m in ("lbr_exact", "simlbr_official"):
            if v not in runs or m not in runs or (v in ("recon", "sup", "rigid-dinov2") and m == "lbr_exact"):
                continue
            name = f"{v}+fuse-{m}"
            for s in (0, 1, 2):
                (d, dg), (l, lg) = runs[v][s], runs[m][s]
                f = zfuse(d, l, rows)
                runs.setdefault(name, {})[s] = (f(d, l), f(dg, lg))
    summ = []
    for m, bys in runs.items():
        for s, (sc, dg) in bys.items():
            out, sm = ap.evaluate(sc, dg, rows)
            sm.update(method=m, seed=s); summ.append(sm)
    sm = pd.DataFrame(summ)
    cols = ["mean_tpr@5%", "worst_tpr@5%", "mean_tpr@1%", "worst_tpr@1%", "fpr_test_reals@5%", "fpr_degraded_reals@5%", "mean_acc"]
    t = sm.groupby("method")[cols].mean().round(1).sort_values("mean_tpr@5%", ascending=False)
    lines = [f"Déjà VAE pilot [{FEAT}{' layers ' + str(MID) if FEAT == 'mid' else ''}], {a.set} ({time.strftime('%Y-%m-%d %H:%M')}); thresholds on the same calibration reals; 3 seeds",
             "comparison heads from " + str(ref) + " (SimLBR preprocessing)", "", t.to_string(), ""]
    per = []
    for m, bys in runs.items():
        for s, (sc, dg) in bys.items():
            out, _ = ap.evaluate(sc, dg, rows)
            for o in out:
                per.append(dict(method=m, seed=s, source=o["source"], tpr5=o["tpr@5%"]))
    pt = pd.DataFrame(per).groupby(["method", "source"])["tpr5"].mean().unstack().round(1)
    if pt.shape[1] > 1:
        lines += ["TPR@5% per generator:", pt.to_string(), ""]
    pairs = [("dv", "dv-single", "H1': relation vs single image (same data)"),
             ("dv", "dv-nocf", "counterfactual reals"), ("dv", "dv-tf", "learned vs training-free"),
             ("dv", "dv-prof", "full relation vs per-layer distance profile"), ("dv-prof", "dv-single", "distance profile vs single image"),
             ("dv", "lbr_exact", ""), ("dv", "simlbr_official", ""),
             ("dv+fuse-lbr_exact", "lbr_exact", "H3': fusion vs SimLBR"),
             ("dv+fuse-lbr_exact", "simlbr_official", ""), ("dv+fuse-lbr_exact", "mix_aligned+lbr", "vs ANCHOR v1 best"),
             ("dv+fuse-simlbr_official", "simlbr_official", "fusion with the released head"),
             ("dv+fuse-simlbr_official", "dv-single+fuse-simlbr_official", "fusion control: relation vs single"),
             ("dv+fuse-simlbr_official", "dv-nocf+fuse-simlbr_official", "fusion control: counterfactuals"),
             ("dv+fuse-simlbr_official", "recon+fuse-simlbr_official", "fusion control: AlignedForensics-style recon"),
             ("dv+fuse-simlbr_official", "sup+fuse-simlbr_official", "fusion control: plain 2-detector ensemble"),
             ("dv-single", "rigid-dinov2", "vs RIGID (published DINOv2)"), ("dv", "rigid-dinov2", "vs RIGID (published DINOv2)"),
             ("dv-single", "rigid-dinov2~norm", "vs RIGID (noise on normalised input; stronger variant)"),
             ("dv-tf", "rigid-dinov3", "training-free vs training-free, same backbone"),
             ("dv+fuse-simlbr_official", "rigid-dinov2+fuse-simlbr_official", "fusion control: RIGID as the partner")]
    lines.append("Paired bootstrap (resample calib reals + test fakes; seed-averaged), TPR@5%:")
    for x, y, note in pairs:
        if x in runs and y in runs:
            r = ap.paired_bootstrap([runs[x][s][0] for s in (0, 1, 2)], [runs[y][s][0] for s in (0, 1, 2)], rows, eps=0.05, B=1000)
            (mm, ml, mh), (wm, wl, wh) = r["mean"], r["worst"]
            lines.append(f"  {x:26s} - {y:18s} mean {mm:+5.1f} [{ml:+.1f},{mh:+.1f}]  worst {wm:+5.1f} [{wl:+.1f},{wh:+.1f}]  {note}")
    txt = "\n".join(lines)
    print(txt); (W / f"report_{a.set}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pairs"); s.add_argument("--set", choices=list(SETS), required=True); s.add_argument("--batch", type=int, default=16)
    t = sub.add_parser("train"); t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--steps", type=int, default=2000); t.add_argument("--batch", type=int, default=500)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--dropout", type=float, default=0.5)
    r = sub.add_parser("report"); r.add_argument("--set", choices=list(SETS), required=True)
    a = p.parse_args()
    dict(pairs=cmd_pairs, train=cmd_train, report=cmd_report)[a.cmd](a)
