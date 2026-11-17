"""Review control for the GenImage decoder bank (Fig. 4): format parity.
dv_bank_full.py copies the ORIGINAL files: GenImage reals are JPEG (ImageNet), fakes PNG. A JPEG real could move more under
some autoencoders for reasons unrelated to generation (e.g. MoVQ quantising JPEG blocking), which would make cross-decoder
detection (MoVQ on SD1 images, SD1 decoder on VQDM) a format effect. Here every view (same images, same 32-px-grid
native crop) is first re-saved as JPEG q95, as in the main GenImage evaluation, and then copied by each autoencoder.
Same DINOv3 layers 10-18, same z-scoring, same in-sample threshold (95th percentile of the 300 reals) as dv_bank_full.
    DV_FEAT=mid anchor/bin/python dv_bank_q95.py   -> runs/dv/bank_q95/report.txt
"""
import os, sys, json, time
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import to256
import dv_bank_full as F

OUT = Path("runs/dv/bank_q95"); OUT.mkdir(parents=True, exist_ok=True)
BANK = {k: v for k, v in F.BANK.items() if k != "vq"}


def main():
    import torch
    G = {k: v for k, v in json.load(open(F.OUT / "groups.json")).items() if k.startswith("gi:")}
    dev = ap.pick_device(); enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

    def cls_layers(ims):
        out = []
        for s in range(0, len(ims), 16):
            with torch.no_grad():
                x = (enc.to_tensor([to256(im) for im in ims[s:s + 16]]) - enc.mean) / enc.std
                hs = enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[L][:, 0] for L in F.LAYERS], 1).float().cpu().numpy())
        return np.concatenate(out)

    views = {k: [ap.jpeg(F.view32(p), 95) for p in v] for k, v in G.items()}
    if not (OUT / "phi_v.npz").exists():
        np.savez(OUT / "phi_v.npz", **{k: cls_layers(v) for k, v in views.items()})
    for name, spec in BANK.items():
        f = OUT / f"phi_R_{name}.npz"
        if f.exists():
            continue
        t0 = time.time(); enc.vae = ap.load_autoencoder(spec, dev); res = {}
        for k, ims in views.items():
            R = [None] * len(ims); by = {}
            for i, im in enumerate(ims):
                by.setdefault(im.size, []).append(i)
            for idx in by.values():
                for i, r in zip(idx, enc.reconstruct([ims[i] for i in idx], 4)):
                    R[i] = r
            res[k] = cls_layers(R)
        np.savez(f, **res); enc.vae = None
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"bank_q95 {name} done in {time.time() - t0:.0f}s", flush=True)
    report()


def report():
    import pandas as pd
    A = dict(np.load(OUT / "phi_v.npz")); names = [n for n in BANK if (OUT / f"phi_R_{n}.npz").exists()]
    R = {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in names}

    def dist(a, b):
        return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)

    def z(n, g):
        d = np.log(dist(A[g], R[n][g]) + 1e-6); r = np.log(dist(A["gi:real"], R[n]["gi:real"]) + 1e-6)
        return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)
    T, U = {}, {}
    for g in [k for k in A if k != "gi:real"]:
        T[g[3:]] = {n: 100 * float((z(n, g) > np.quantile(z(n, "gi:real"), .95)).mean()) for n in names}
        U[g[3:]] = {n: F.auroc(z(n, g), z(n, "gi:real")) for n in names}
    txt = "\n".join(["GenImage decoder bank with JPEG q95 on every view before the copy (format parity control).",
                     "TPR@5% (threshold: 95th pct of the 300 reals)", pd.DataFrame(T).T.round(1).to_string(), "",
                     "AUROC", pd.DataFrame(U).T.round(3).to_string()])
    print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main() if len(sys.argv) < 2 else report()
