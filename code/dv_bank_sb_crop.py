"""Review control for E5b: the Synthbuster decoder bank with CROP-matched reals (synthbuster_crop.py) instead of the
downscale-matched reals. Same 200 fakes per generator and the same RAISE ids as dv_bank_sb.py (groups.json); only the
reals change, to a native-resolution centre crop of the RAISE photograph at the fake's exact size (no resampling).
Fake features are reused from runs/sb/bank. Generators: the five with a pre-specified own decoder (SD 1.3, 1.4, 2, XL,
DALL-E 3). Same layers, z-scoring and in-sample threshold (95th percentile of the 200 reals) as dv_bank_sb.py.
    DV_FEAT=mid anchor/bin/python dv_bank_sb_crop.py run      # -> runs/sb/bank_crop/*.npz, report.txt
"""
import os, sys, json, time
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import to256
import dv_bank_sb as B
from dv_bank_full import view32, auroc, LAYERS

SRC = Path("runs/sb/bank"); OUT = Path("runs/sb/bank_crop"); OUT.mkdir(parents=True, exist_ok=True)
CROP = Path.home() / "anchor_data" / "synthbuster_real_cropped"
GENS = list(B.OWN)


def groups():
    G = json.load(open(SRC / "groups.json"))
    return {f"{g}:real": [str(CROP / g / (Path(p).stem + ".png")) for p in G[f"{g}:real"]] for g in GENS}


def cmd_run():
    import torch
    G = groups(); dev = ap.pick_device(); enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

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
    for name, spec in B.BANK.items():
        f = OUT / f"phi_R_{name}.npz"
        if f.exists():
            continue
        t0 = time.time(); enc.vae = ap.load_autoencoder(spec, dev); res = {}
        cd = name == "cdec"
        if cd and dev.type == "mps":
            enc.vae = enc.vae.half()
        for k, ims in views.items():
            R = [None] * len(ims); by = {}
            for i, im in enumerate(ims):
                by.setdefault(im.size, []).append(i)
            for idx in by.values():
                for i, r in zip(idx, B.reconstruct(enc, [ims[i] for i in idx], cdec=cd, vae_batch=1 if cd else 4)):
                    R[i] = r
            res[k] = cls_layers(R)
            if dev.type == "mps":
                torch.mps.empty_cache()
        np.savez(f, **res); enc.vae = None
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"bank_crop {name:7s} done in {time.time() - t0:5.0f}s", flush=True)
    cmd_report()


def cmd_report():
    import pandas as pd
    A0, R0, names = B.load()
    Ac = dict(np.load(OUT / "phi_v.npz")); names = [n for n in names if (OUT / f"phi_R_{n}.npz").exists()]
    Rc = {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in names}
    A = {**A0, **{k.replace(":real", ":crop"): v for k, v in Ac.items()}}
    R = {n: {**R0[n], **{k.replace(":real", ":crop"): v for k, v in Rc[n].items()}} for n in names}
    T, U, T0 = {}, {}, {}
    for g in GENS:
        for tag, ref, D in (("crop", f"{g}:crop", T), ("orig", f"{g}:real", T0)):
            Sf = np.stack([B.zscore(A, R, k, f"{g}:fake", ref) for k in names], 1)
            Sr = np.stack([B.zscore(A, R, k, ref, ref) for k in names], 1)
            D[g] = {k: 100 * (Sf[:, j] > np.quantile(Sr[:, j], 0.95)).mean() for j, k in enumerate(names)}
            if tag == "crop":
                U[g] = {k: auroc(Sf[:, j], Sr[:, j]) for j, k in enumerate(names)}
    L = ["E5b with CROP-matched reals (native-resolution crops, no resampling); same 200 fakes per generator.",
         "Threshold: 95th percentile of the generator's 200 crop-matched reals. sdxl10 is post hoc.", "",
         "TPR@5% (%), crop-matched reals", pd.DataFrame(T).T.round(1).to_string(), "",
         "TPR@5% (%), original downscale-matched reals (same columns, for comparison)", pd.DataFrame(T0).T.round(1).to_string(), "",
         "AUROC, crop-matched reals", pd.DataFrame(U).T.round(3).to_string(), "", "Own decoder = row maximum among single AEs?"]
    for g, k in B.OWN.items():
        if k in T[g]:
            row = {a: T[g][a] for a in names if a != "sdxl10"}; best = max(row, key=row.get)
            L.append(f"  {g:22s} own={k:7s} TPR {row[k]:5.1f}  row max={best} ({row[best]:.1f})  -> {'HOLDS' if best == k else 'FAILS'}"
                     + (f"   (sdxl10: {T[g]['sdxl10']:.1f})" if "sdxl10" in T[g] else ""))
    txt = "\n".join(L); print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
