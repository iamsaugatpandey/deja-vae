"""E1 + E2 (full n): WHERE does the autoencoder fixed point live, in which backbone, and does it need native resolution?

Training-free. For each image view v and its autoencoder copy R(v) (SD ft-MSE autoencoder, as in Déjà VAE), the
cosine distance between v and R(v) is measured at every 2nd layer of two frozen backbones:
    dinov3  DINOv3 ViT-L/16 (SimLBR normalisation, 256 px input)    CLS + mean patch distance
    dinov2  DINOv2 ViT-L/14 (ImageNet normalisation, 224 px input)  CLS + mean patch distance
plus PSNR(v, R(v)) as the pixel baseline. Score = -distance (moved less = more likely generated).

Conditions (E2 = how the copy is made):
    raw    native-resolution centre crop (<= 256 px, 8-px grid)  -> R      (Déjà VAE's choice)
    jpeg   raw, then JPEG q95 on both v and R(v)                           (the p2 protocol)
    x0.75  image downscaled x0.75 first, then the native crop -> R         (one resize after generation)
    sq     image squashed to 256 x 256 first (SimLBR preprocessing) -> R   (copy made after resizing)
Rows (seed 7): GenImage 400 eval reals + 200 per generator (8 generators); Chameleon 300 + 300 (raw only).
Real rows also get the training-row measurements (simulated fake R(x), blur-matched counterfactual) under raw.
Every distance is saved per image, so figures can bootstrap anything.

    anchor/bin/python dv_layers_full.py run       # resumable per (group, condition); ~1.5 h on the Mac
    anchor/bin/python dv_layers_full.py report    # -> runs/dv/layers_full/report.txt (AUROC + 95% bootstrap CI)
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import view

OUT = Path("runs/dv/layers_full"); OUT.mkdir(parents=True, exist_ok=True)
LAYERS = list(range(2, 25, 2))
N_REAL, N_GEN, N_CHAM = 400, 200, 300
CONDS_GI = ["raw", "jpeg", "x0.75", "sq"]
BACKBONES = {"dinov3": (ap.DINO_ID, ap.SIMLBR_MEAN, ap.SIMLBR_STD, 256),
             "dinov2": ("facebook/dinov2-large", ap.IMNET_MEAN, ap.IMNET_STD, 224)}


def jpeg95(im):
    return ap.jpeg(im, 95)


def make_view(path, cond):
    from PIL import Image
    if cond in ("raw", "jpeg"):
        return view(path)
    im = Image.open(path).convert("RGB")
    if cond == "x0.75":
        im = im.resize((max(8, round(im.width * 0.75)), max(8, round(im.height * 0.75))), Image.BICUBIC)
    elif cond == "sq":
        im = im.resize((256, 256), Image.BICUBIC)
    w, h = im.size
    cw, ch = min(256, w // 8 * 8), min(256, h // 8 * 8)
    l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return im.crop((l, t, l + cw, t + ch))


class Probe:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.t = torch
        self.enc = ap.Encoders(ap.pick_device(), need_vae=True, norm="simlbr")   # ft-MSE autoencoder + DINOv3
        dev = self.enc.device
        self.models = {}
        for name, (mid, mean, std, size) in BACKBONES.items():
            m = self.enc.dino if name == "dinov3" else AutoModel.from_pretrained(mid).to(dev).eval()
            skip = 1 + int(getattr(m.config, "num_register_tokens", 0) or 0)
            self.models[name] = (m, torch.tensor(mean, device=dev).view(1, 3, 1, 1),
                                 torch.tensor(std, device=dev).view(1, 3, 1, 1), size, skip)

    def tokens(self, name, ims):
        from PIL import Image
        m, M, S, size, _ = self.models[name]
        ims = [im if im.size == (size, size) else im.resize((size, size), Image.BICUBIC) for im in ims]
        with self.t.no_grad():
            x = (self.enc.to_tensor(ims) - M) / S
            hs = m(pixel_values=x, output_hidden_states=True).hidden_states
            return {L: hs[L].float() for L in LAYERS}

    def measure(self, A, B):
        res = {}
        for s in range(0, len(A), 8):
            for name in self.models:
                skip = self.models[name][4]
                ta, tb = self.tokens(name, A[s:s + 8]), self.tokens(name, B[s:s + 8])
                for L in LAYERS:
                    d = (1 - self.t.nn.functional.cosine_similarity(ta[L], tb[L], dim=-1)).cpu().numpy()
                    res.setdefault(f"{name}/cls@{L}", []).extend(d[:, 0])
                    res.setdefault(f"{name}/pmean@{L}", []).extend(d[:, skip:].mean(1))
        res["psnr"] = [-ap.psnr(a, b) for a, b in zip(A, B)]
        return {k: np.asarray(v, np.float32) for k, v in res.items()}

    def recon(self, ims):
        out = [None] * len(ims); by = {}
        for i, im in enumerate(ims):
            by.setdefault(im.size, []).append(i)
        for idx in by.values():
            for i, r in zip(idx, self.enc.reconstruct([ims[i] for i in idx], 4)):
                out[i] = r
        return out


def groups():
    rng = np.random.default_rng(7); g = {}
    rows = ap.read_manifest(Path("runs/p2/manifest.csv")); ev = [r for r in rows if r["split"] != "train"]
    reals = [r["path"] for r in ev if r["label"] == 0]
    g["gi:real"] = [reals[i] for i in rng.choice(len(reals), N_REAL, replace=False)]
    for src in sorted({r["source"] for r in ev if r["label"] == 1}):
        c = [r["path"] for r in ev if r["label"] == 1 and r["source"] == src]
        g[f"gi:{src}"] = [c[i] for i in rng.choice(len(c), min(N_GEN, len(c)), replace=False)]
    cr = ap.read_manifest(Path("runs/cham/manifest.csv")); ce = [r for r in cr if r["split"] != "train"]
    for lab, nm in ((0, "real"), (1, "fake")):
        c = [r["path"] for r in ce if r["label"] == lab]
        g[f"cham:{nm}"] = [c[i] for i in rng.choice(len(c), N_CHAM, replace=False)]
    return g


def cmd_run():
    P = Probe(); G = groups()
    json.dump(G, open(OUT / "groups.json", "w"))
    for name, paths in G.items():
        for cond in (CONDS_GI if name.startswith("gi:") else ["raw"]):
            f = OUT / f"{name.replace(':', '_')}_{cond}.npz"
            if f.exists():
                continue
            t0 = time.time()
            ims = [make_view(p, cond) for p in paths]
            R = P.recon(ims)
            if cond == "jpeg":
                ims, R = [jpeg95(i) for i in ims], [jpeg95(r) for r in R]
            out = {f"x/{k}": v for k, v in P.measure(ims, R).items()}
            if name == "gi:real" and cond == "raw":
                RR = P.recon(R)
                out.update({f"sim/{k}": v for k, v in P.measure(R, RR).items()})
                D = [ap.blur_to_psnr(a, ap.psnr(a, b)) for a, b in zip(ims, R)]
                out.update({f"blur/{k}": v for k, v in P.measure(D, P.recon(D)).items()})
            np.savez(f, **out)
            print(f"{name:32s} {cond:5s} n={len(ims)}  {time.time() - t0:5.0f}s", flush=True)
    print("LAYERS FULL RUN DONE", flush=True)
    cmd_report()


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([-pos, -neg])); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auroc(pos, neg, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def cmd_report():
    import pandas as pd
    L = lambda n, c: dict(np.load(OUT / f"{n}_{c}.npz"))
    gens = sorted(f.stem[3:-4] for f in OUT.glob("gi_*_raw.npz") if f.stem != "gi_real_raw")
    lines = ["E1/E2 layer diagnostic (full n): AUROC generated vs real, score = -(distance between v and R(v)).",
             "MEAN = mean over GenImage generators; [95% CI] = bootstrap over images of the MEAN-row's best layer.", ""]
    for cond in CONDS_GI:
        if not (OUT / f"gi_real_{cond}.npz").exists():
            continue
        real = L("gi_real", cond); keys = [k[2:] for k in real if k.startswith("x/")]
        tab = {}
        for k in keys:
            row = {g.replace("stable_diffusion_v_1_", "SD1."): auroc(L(f"gi_{g}", cond)[f"x/{k}"], real[f"x/{k}"])
                   for g in gens if (OUT / f"gi_{g}_{cond}.npz").exists()}
            row["MEAN"] = np.mean(list(row.values()))
            if cond == "raw":
                row["simR(x)"] = auroc(real[f"sim/{k}"], real[f"x/{k}"]); row["blur*"] = auroc(real[f"blur/{k}"], real[f"x/{k}"])
                if (OUT / "cham_fake_raw.npz").exists():
                    row["CHAM"] = auroc(L("cham_fake", "raw")[f"x/{k}"], L("cham_real", "raw")[f"x/{k}"])
            tab[k] = row
        df = pd.DataFrame(tab).T.round(3)
        lines += [f"== condition: {cond}", df.sort_index().to_string(), ""]
        for bb in BACKBONES:
            sub = df[[i.startswith(bb + "/cls") for i in df.index]]
            if len(sub):
                best = sub["MEAN"].idxmax()
                lines.append(f"   {bb}: best CLS layer {best} MEAN {sub.loc[best, 'MEAN']:.3f}; final layer {bb}/cls@24 MEAN {df.loc[bb + '/cls@24', 'MEAN']:.3f}")
        lines.append("")
    txt = "\n".join(lines); print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
