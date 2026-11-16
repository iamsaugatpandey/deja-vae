"""Déjà VAE: does the fixed-point signal on Chameleon depend on WHICH autoencoder makes the copy?
v1-v3 only used the SD-1.x VAE. If Chameleon fakes come from SDXL / FLUX / other latent spaces, a copy
with the SD-1 VAE shows no "already copied" effect.

Training-free, no heads: per image view v (native crop, 32-px grid so DC-AE fits), per autoencoder k,
distance d_k = cosine distance of DINOv3 CLS at layers {10, 12, 14, 16, 18} between v and R_k(v),
summed after z-scoring on the reals. Score = -d (moved less = more likely generated).
Reports AUROC and TPR@5% (threshold from the same set's reals) per AE, the bank max
(panel b: max_k of z-scored scores), and Chameleon groups (raw-looking %64 vs resized, PNG).

    anchor/bin/python dv_bank.py run | report        -> runs/dv/bank/report.txt
"""
import os, sys, json, time
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import to256

OUT = Path("runs/dv/bank"); OUT.mkdir(parents=True, exist_ok=True)
BANK = {"sd1": "stabilityai/sd-vae-ft-mse", "sdxl": "stabilityai/sdxl-vae", "flux": "ostris/Flex.1-alpha:vae",
        "vq": "kandinsky-community/kandinsky-2-2-decoder:movq", "dcae": "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers"}
LAYERS = [10, 12, 14, 16, 18]
N_CR, N_CF, N_GR, N_GF = 300, 300, 150, 60


def view32(path):
    from PIL import Image
    im = Image.open(path).convert("RGB"); w, h = im.size
    cw, ch = min(256, w // 32 * 32), min(256, h // 32 * 32)
    l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return im.crop((l, t, l + cw, t + ch))


def groups():
    rng = np.random.default_rng(5)
    g = {}
    c = ap.read_manifest(Path("runs/cham/manifest.csv")); ce = [r for r in c if r["split"] == "test"]
    for lab, nm, n in ((0, "cham:real", N_CR), (1, "cham:fake", N_CF)):
        p = [r["path"] for r in ce if r["label"] == lab]
        g[nm] = [p[i] for i in rng.choice(len(p), n, replace=False)]
    r = ap.read_manifest(Path("runs/p2/manifest.csv")); ev = [x for x in r if x["split"] != "train"]
    p = [x["path"] for x in ev if x["label"] == 0]; g["gi:real"] = [p[i] for i in rng.choice(len(p), N_GR, replace=False)]
    for src in ("stable_diffusion_v_1_4", "Midjourney", "VQDM", "ADM"):
        p = [x["path"] for x in ev if x["label"] == 1 and x["source"] == src]
        g[f"gi:{src}"] = [p[i] for i in rng.choice(len(p), N_GF, replace=False)]
    return g


def cmd_run():
    import torch
    G = groups(); json.dump(G, open(OUT / "groups.json", "w"))
    dev = ap.pick_device()
    enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

    def cls_layers(ims):
        out = []
        for s in range(0, len(ims), 16):
            with torch.no_grad():
                x = (enc.to_tensor([to256(im) for im in ims[s:s + 16]]) - enc.mean) / enc.std
                hs = enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[L][:, 0] for L in LAYERS], 1).float().cpu().numpy())
        return np.concatenate(out)

    views = {k: [view32(p) for p in v] for k, v in G.items()}
    fa = OUT / "phi_v.npz"
    if not fa.exists():
        np.savez(fa, **{k: cls_layers(v) for k, v in views.items()})
    for name, spec in BANK.items():
        f = OUT / f"phi_R_{name}.npz"
        if f.exists():
            continue
        t0 = time.time()
        enc.vae = ap.load_autoencoder(spec, dev)
        res = {}
        for k, ims in views.items():
            R = [None] * len(ims); by = {}
            for i, im in enumerate(ims):
                by.setdefault(im.size, []).append(i)
            for idx in by.values():
                for i, r in zip(idx, enc.reconstruct([ims[i] for i in idx], 4)):
                    R[i] = r
            res[k] = cls_layers(R)
        np.savez(f, **res)
        enc.vae = None; torch.mps.empty_cache() if dev.type == "mps" else None
        print(f"bank {name:5s} done in {time.time() - t0:5.0f}s", flush=True)
    sizes = {}
    from PIL import Image
    for k in ("cham:fake",):
        sizes[k] = []
        for p in G[k]:
            with Image.open(p) as im:
                sizes[k].append([im.size[0], im.size[1], im.format or "?"])
    json.dump(sizes, open(OUT / "sizes.json", "w"))
    print("BANK RUN DONE", flush=True)
    cmd_report()


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg])); n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_report():
    import pandas as pd
    A = dict(np.load(OUT / "phi_v.npz"))
    names = [n for n in BANK if (OUT / f"phi_R_{n}.npz").exists()]
    Rk = {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in names}
    G = json.load(open(OUT / "groups.json")); sz = json.load(open(OUT / "sizes.json"))["cham:fake"]
    m64 = np.array([w % 64 == 0 and h % 64 == 0 for w, h, _ in sz]); n8 = np.array([w % 8 != 0 or h % 8 != 0 for w, h, _ in sz])
    png = np.array([f == "PNG" for _, _, f in sz])

    def dist(a, b):   # [n, L, D] -> per-layer cosine distance [n, L]
        return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)

    def score(n, grp, ref):   # z-score each layer's log-distance on the reference reals, sum; higher = more generated
        d = np.log(dist(A[grp], Rk[n][grp]) + 1e-6); r = np.log(dist(A[ref], Rk[n][ref]) + 1e-6)
        return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)

    lines = ["Training-free fixed-point score per autoencoder (DINOv3 CLS layers 10-18, z-scored on the set's reals).",
             "AUROC (0.5 = none, >0.5 = generated images move less under that AE) and TPR@5% (threshold: the set's reals).", ""]
    tab = []
    sets = [("Chameleon", "cham:real", {"all": "cham:fake"}), ("GenImage", "gi:real", {g: g for g in G if g.startswith("gi:") and g != "gi:real"})]
    for n in names + ["BANK max", "BANK mean"]:
        row = {"AE": n}
        for sname, ref, fakes in sets:
            def S(grp):
                if n == "BANK max":
                    return np.max([score(k, grp, ref) for k in names], 0)
                if n == "BANK mean":
                    return np.mean([score(k, grp, ref) for k in names], 0)
                return score(n, grp, ref)
            sr = S(ref); thr = np.quantile(sr, 0.95)
            for lab, grp in fakes.items():
                sf = S(grp)
                short = lab.replace("gi:", "").replace("stable_diffusion_v_1_4", "SD1.4")
                key = "CHAM" if sname == "Chameleon" else short
                row[f"{key} AUC"] = round(auroc(sf, sr), 3); row[f"{key} TPR"] = round(100 * float((sf > thr).mean()), 1)
                if sname == "Chameleon":
                    for gname, m in (("%64", m64), ("resized", n8), ("PNG", png)):
                        row[f"CHAM {gname} TPR"] = round(100 * float((sf[m] > thr).mean()), 1)
        tab.append(row)
    df = pd.DataFrame(tab).set_index("AE")
    lines += ["Chameleon (n_real=%d, n_fake=%d; %%64 n=%d, resized n=%d, PNG n=%d)" % (N_CR, N_CF, m64.sum(), n8.sum(), png.sum()),
              df[[c for c in df.columns if c.startswith("CHAM")]].to_string(), "",
              "GenImage (n_real=%d, %d per generator; no JPEG re-save here)" % (N_GR, N_GF),
              df[[c for c in df.columns if not c.startswith("CHAM")]].to_string(), ""]
    # which AE gives each Chameleon fake its strongest fixed-point score
    Z = np.stack([score(k, "cham:fake", "cham:real") for k in names], 1)
    best = pd.Series(np.array(names)[Z.argmax(1)]).value_counts(normalize=True).round(3) * 100
    lines += ["Chameleon fakes: share (%) whose strongest z-scored fixed-point score comes from each AE:", best.to_string()]
    txt = "\n".join(lines); print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
