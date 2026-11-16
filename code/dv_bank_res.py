"""E4 control: is the bank's BigGAN result a resolution artefact?

GenImage BigGAN images are 128 x 128; every other GenImage group has short side >= 256, and the 300 GenImage reals
have median short side 375. The copy is made from a native crop (<= 256 px), so BigGAN copies are made at 128 px and
then upsampled for the backbone. Control: the SAME 300 GenImage reals, downscaled (bicubic, antialiased) to short side
128 and cropped exactly like the others -> "gi:real128". If real128 already looks "generated" under an AE (AUROC vs.
native reals >> 0.5), the AE's BigGAN signal is resolution, not generation. We also re-threshold BigGAN on real128.
Uses runs/dv/bank_full features for the native groups.  Output: runs/dv/bank_full/res_control.txt
    anchor/bin/python dv_bank_res.py
"""
import os, sys, json
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import to256
import dv_bank_full as B

OUT = B.OUT; F = OUT / "res128.npz"
NAMES = [n for n in B.BANK if n != "vq"]


def views128():
    from PIL import Image
    G = json.load(open(OUT / "groups.json")); vs = []
    for p in G["gi:real"]:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = 128 / min(w, h)
        im = im.resize((max(128, round(w * s)), max(128, round(h * s))), Image.BICUBIC)
        w, h = im.size; l, t = (w - 128) // 2, (h - 128) // 2
        vs.append(im.crop((l, t, l + 128, t + 128)))
    return vs


def run():
    import torch
    dev = ap.pick_device(); enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

    def cls_layers(ims):
        out = []
        for s in range(0, len(ims), 16):
            with torch.no_grad():
                x = (enc.to_tensor([to256(im) for im in ims[s:s + 16]]) - enc.mean) / enc.std
                hs = enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[L][:, 0] for L in B.LAYERS], 1).float().cpu().numpy())
        return np.concatenate(out)

    v = views128(); res = {"phi_v": cls_layers(v)}
    for n in NAMES:
        enc.vae = ap.load_autoencoder(B.BANK[n], dev)
        res[n] = cls_layers(list(enc.reconstruct(v, 4))); enc.vae = None
        print(n, "done", flush=True)
    np.savez(F, **res)


def report():
    A = dict(np.load(OUT / "phi_v.npz")); R = {n: dict(np.load(OUT / f"phi_R_{n}.npz")) for n in NAMES}; C = dict(np.load(F))

    def dist(a, b):
        return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)

    def raw(n, grp):
        return np.log(dist(C["phi_v"], C[n]) + 1e-6) if grp == "real128" else np.log(dist(A[grp], R[n][grp]) + 1e-6)

    def score(n, grp, ref):
        d, r = raw(n, grp), raw(n, ref)
        return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)

    L = ["Resolution control (300 GenImage reals downscaled to 128 px = real128).",
         "AUC(real128 vs native reals): >0.5 means downscaled REAL images already look 'generated' under that AE.",
         "BigGAN TPR@5%: threshold from native reals (as in the bank table) vs from real128 (resolution-matched).", "",
         f"{'AE':8s} {'AUC real128':>12s} {'BigGAN AUC/TPR (native ref)':>28s} {'BigGAN AUC/TPR (real128 ref)':>29s}"]
    for n in NAMES:
        a = B.auroc(score(n, "real128", "gi:real"), score(n, "gi:real", "gi:real"))
        out = []
        for ref in ("gi:real", "real128"):
            sr, sf = score(n, ref, ref), score(n, "gi:BigGAN", ref)
            out.append(f"{B.auroc(sf, sr):.3f} / {100 * (sf > np.quantile(sr, 0.95)).mean():5.1f}")
        L.append(f"{n:8s} {a:12.3f} {out[0]:>28s} {out[1]:>29s}")
    txt = "\n".join(L); print(txt); (OUT / "res_control.txt").write_text(txt + "\n")


if __name__ == "__main__":
    if not F.exists():
        run()
    report()
