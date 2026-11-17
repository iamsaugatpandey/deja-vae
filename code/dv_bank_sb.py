"""E5b: decoder-matching test on Synthbuster (training-free autoencoder bank). Predictions stated before the run (paper, Sec. 8).

Per Synthbuster generator: 200 fakes and their 200 geometry-matched TEST reals (the same RAISE ids; synthbuster_prep.py),
so reals and fakes have identical sizes and resolution cannot drive the result. View = native centre crop <= 256 px on
the 32-px grid (as dv_bank_full.py), no JPEG re-save. Score per autoencoder k: cosine distances of DINOv3 CLS at layers
10-18 between v and R_k(v), log, z-scored per layer on that generator's reals, summed; higher = moved less.
Bank = dv_bank_full.BANK minus the duplicate MoVQ, plus OpenAI's consistency decoder (SD encoder, own diffusion decoder,
2 steps, fixed seed) -- the decoder that the DALL-E 3 report says DALL-E 3 decodes with.

    anchor/bin/python dv_bank_sb.py run      # -> runs/sb/bank/*.npz, then report
    anchor/bin/python dv_bank_sb.py report   # -> runs/sb/bank/report.txt
"""
import os, sys, json, time
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import to256
from dv_bank_full import view32, auroc, LAYERS

OUT = Path("runs/sb/bank"); OUT.mkdir(parents=True, exist_ok=True)
N = 200
BANK = {"sd1orig": "stable-diffusion-v1-5/stable-diffusion-v1-5:vae", "ftmse": "stabilityai/sd-vae-ft-mse",
        "sdxl": "stabilityai/sdxl-vae", "kd21": "kandinsky-community/kandinsky-2-1:movq", "flux": "ostris/Flex.1-alpha:vae",
        "dcae": "mit-han-lab/dc-ae-f32c32-in-1.0-diffusers", "cdec": "openai/consistency-decoder",
        "sdxl10": "stabilityai/stable-diffusion-xl-base-1.0:vae_1_0"}   # sdxl10: post hoc (dv_bank_sdxl.py); not pre-specified
OWN = {"stable-diffusion-1-3": "sd1orig", "stable-diffusion-1-4": "sd1orig", "stable-diffusion-2": "ftmse",
       "stable-diffusion-xl": "sdxl", "dalle3": "cdec"}


def groups():
    rows = ap.read_manifest(Path("runs/sb/manifest.csv")); rng = np.random.default_rng(5); G = {}
    for g in sorted({r["source"] for r in rows}):
        reals = {Path(r["path"]).stem: r["path"] for r in rows if r["source"] == g and r["label"] == 0 and r["split"] == "test"}
        fakes = {Path(r["path"]).stem: r["path"] for r in rows if r["source"] == g and r["label"] == 1}
        ids = sorted(set(reals) & set(fakes)); pick = [ids[k] for k in rng.choice(len(ids), N, replace=False)]
        G[f"{g}:real"] = [reals[i] for i in pick]; G[f"{g}:fake"] = [fakes[i] for i in pick]
    return G


def reconstruct(enc, ims, cdec=False, vae_batch=4):
    """anchor_pilot.Encoders.reconstruct, plus a fixed generator for the (stochastic) consistency decoder."""
    from PIL import Image
    t = enc.torch; out = []
    with t.no_grad():
        for s in range(0, len(ims), vae_batch):
            x = (enc.to_tensor(ims[s:s + vae_batch]) * 2 - 1).to(enc.vae.dtype)
            e = enc.vae.encode(x)
            z = e.latent_dist.mode() if hasattr(e, "latent_dist") else (e.latents if hasattr(e, "latents") else e.latent)
            if cdec:
                y = enc.vae.decode(z.contiguous(), generator=t.Generator().manual_seed(0)).sample
            else:
                y = enc.vae.decode(z.contiguous()).sample
            y = ((y.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            out += [Image.fromarray(v) for v in y]
            del x, e, z
            if cdec and enc.device.type == "mps":
                t.mps.empty_cache()
    return out


def cmd_run():
    import torch
    G = groups(); json.dump(G, open(OUT / "groups.json", "w"))
    dev = ap.pick_device(); enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

    def cls_layers(ims):
        out = []
        for s in range(0, len(ims), 16):
            with torch.no_grad():
                x = (enc.to_tensor([to256(im) for im in ims[s:s + 16]]) - enc.mean) / enc.std
                hs = enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[L][:, 0] for L in LAYERS], 1).float().cpu().numpy())
        return np.concatenate(out)

    views = {k: [view32(p) for p in v] for k, v in G.items()}
    if not (OUT / "phi_v.npz").exists():
        np.savez(OUT / "phi_v.npz", **{k: cls_layers(v) for k, v in views.items()})
    for name, spec in BANK.items():
        f = OUT / f"phi_R_{name}.npz"
        if f.exists():
            continue
        t0 = time.time(); enc.vae = ap.load_autoencoder(spec, dev); res = {}
        cd = name == "cdec"   # pixel-space diffusion decoder: batch 1, fp16, cache release; fp32/batch 4 spilled 16 GB to swap (16 Nov)
        if cd and dev.type == "mps":
            enc.vae = enc.vae.half()
        for gi, (k, ims) in enumerate(views.items()):
            R = [None] * len(ims); by = {}
            for i, im in enumerate(ims):
                by.setdefault(im.size, []).append(i)
            for idx in by.values():
                for i, r in zip(idx, reconstruct(enc, [ims[i] for i in idx], cdec=cd, vae_batch=1 if cd else 4)):
                    R[i] = r
            res[k] = cls_layers(R)
            if dev.type == "mps":
                torch.mps.empty_cache()
            print(f"  {name} {gi + 1}/{len(views)} {k}  {time.time() - t0:5.0f}s", flush=True)
        np.savez(f, **res); enc.vae = None
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"bank {name:7s} done in {time.time() - t0:5.0f}s", flush=True)
    print("SB BANK RUN DONE", flush=True)
    cmd_report()


def load():
    A = dict(np.load(OUT / "phi_v.npz")); names = [n for n in BANK if (OUT / f"phi_R_{n}.npz").exists()]
    return A, {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in names}, names


def zscore(A, R, k, grp, ref):
    def dist(a, b):
        return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)
    d, r = np.log(dist(A[grp], R[k][grp]) + 1e-6), np.log(dist(A[ref], R[k][ref]) + 1e-6)
    return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)


def cmd_report():
    import pandas as pd
    A, R, names = load(); gens = sorted({k.split(":")[0] for k in A})
    T, U, att = {}, {}, {}
    for g in gens:
        Sf = np.stack([zscore(A, R, k, f"{g}:fake", f"{g}:real") for k in names], 1)
        Sr = np.stack([zscore(A, R, k, f"{g}:real", f"{g}:real") for k in names], 1)
        T[g] = {k: 100 * (Sf[:, j] > np.quantile(Sr[:, j], 0.95)).mean() for j, k in enumerate(names)}
        U[g] = {k: auroc(Sf[:, j], Sr[:, j]) for j, k in enumerate(names)}
        T[g]["BANK max"] = 100 * (Sf.max(1) > np.quantile(Sr.max(1), 0.95)).mean(); U[g]["BANK max"] = auroc(Sf.max(1), Sr.max(1))
        att[g] = pd.Series(np.array(names)[Sf.argmax(1)]).value_counts(normalize=True).reindex(names).fillna(0) * 100
    L = ["E5b Synthbuster decoder bank (training-free; 200 fakes + 200 geometry-matched test reals per generator).",
         "Threshold: 95th percentile of the generator's own matched reals. * = the generator's own decoder (pre-specified).", "",
         "TPR@5% (%)", pd.DataFrame(T).T.round(1).to_string(), "", "AUROC", pd.DataFrame(U).T.round(3).to_string(), "",
         "Attribution: share (%) of each generator's fakes whose strongest score comes from each AE",
         pd.DataFrame(att).T.round(1).to_string(), "", "Pre-specified checks (own decoder = row maximum among single AEs?):"]
    for g, k in OWN.items():
        if g in T and k in T[g]:
            row = {a: T[g][a] for a in names}; best = max(row, key=row.get)
            L.append(f"  {g:22s} own={k:7s} TPR {row[k]:5.1f}  row max={best} ({row[best]:.1f})  -> {'HOLDS' if best == k else 'FAILS'}")
    txt = "\n".join(L); print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
