"""Review: does a mild blur hide a generated image? (evasion boundary of the blur inversion in Sec. 4)
Degraded REAL test images are blurred with sigma ~ U(0.5, 1.5) (Sec. 3). Here the SAME blur is applied to generated test
images, through each method's own degraded-image pipeline, and the threshold stays where the calibration reals put it:
  Deja VAE (I / P / TF): native crop -> blur -> JPEG q95 -> copy -> DINOv3 layers 8-18 (exactly the `deg` path of dejavae pairs)
  SimLBR (released):    squash to 256 -> blur -> JPEG q95 -> DINOv3 final CLS (exactly the cls_degraded path of features)
200 fakes per GenImage generator (seed 3). Reports TPR@5% on the same fakes unblurred (existing scores) and blurred.
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python dv_evasion.py   -> runs/dv2/evasion.txt
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv


def main():
    import torch
    from PIL import ImageFilter
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    rows, idx = dv.rows_of("genimage"); N = len(rows)
    lab, split, src = ap.split_idx(rows)
    rng = np.random.default_rng(3); sig = np.random.default_rng(11).uniform(0.5, 1.5, N)
    pick = np.concatenate([rng.choice(np.where((split == "test") & (lab == 1) & (src == g))[0], 200, replace=False)
                           for g in sorted(set(src[(split == "test") & (lab == 1)]))])
    enc = ap.Encoders(ap.pick_device(), norm="simlbr", vae=dv.AE); P = dv.Pairs(enc)
    A, B, C = [], [], []
    for s in range(0, len(pick), 16):
        b = pick[s:s + 16]
        x = [ap.jpeg(dv.view(rows[i]["path"]).filter(ImageFilter.GaussianBlur(float(sig[i]))), 95) for i in b]
        A.append(P.phi(x)); B.append(P.phi(P.recon(x)))
        sq = [ap.jpeg(ap.load_rgb(rows[i]["path"], 256, "simlbr").filter(ImageFilter.GaussianBlur(float(sig[i]))), 95) for i in b]
        C.append(enc.cls(sq))
        if s % 320 == 0:
            print(f"{s}/{len(pick)}", flush=True)
    A, B, C = np.concatenate(A), np.concatenate(B), np.concatenate(C)
    S = dv.scores_for("genimage", rows, idx)                     # unblurred scores (all rows), per variant and seed
    cal = (split == "calib") & (lab == 0)
    lines = ["Evasion: TPR@5% on 1,600 GenImage fakes (200/generator), unblurred vs blurred with the degraded-real blur.",
             "Threshold: split conformal on the calibration reals (unchanged)."]
    g_of = src[pick]
    def report(name, clean, blurred):
        t = [ap.conformal_tau(c[cal], 0.05) for c in clean]
        tc = np.mean([(c[pick] > ti).mean() for c, ti in zip(clean, t)]) * 100
        tb = np.mean([(bl > ti).mean() for bl, ti in zip(blurred, t)]) * 100
        per = {g: np.mean([(bl[g_of == g] > ti).mean() for bl, ti in zip(blurred, t)]) * 100 for g in sorted(set(g_of))}
        lines.append(f"{name:18s} unblurred {tc:5.1f}  blurred {tb:5.1f}   per generator (blurred): " +
                     ", ".join(f"{g[:12]} {v:.0f}" for g, v in per.items()))
    for v, (mode, _) in dv.VARIANTS.items():
        if v not in ("dv-single", "dv"):
            continue
        heads = [ap.head_from_state(torch.load(dv.W / f"head_{v}_seed{s}.pt", map_location="cpu")) for s in (0, 1, 2)]
        X = dv.rel(A, B, mode)
        with torch.no_grad():
            bl = [h(torch.from_numpy(X.astype(np.float32))).squeeze(1).numpy() for h in heads]
        report({"dv-single": "Deja VAE-I", "dv": "Deja VAE-P"}[v], [S[v][s][0] for s in (0, 1, 2)], bl)
    j = dv.MID.index(dv.TF_LAYER); a, b = A[:, j * 1024:(j + 1) * 1024].astype(np.float32), B[:, j * 1024:(j + 1) * 1024].astype(np.float32)
    tf = -np.log(np.clip(1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), 1e-6, None))
    report("Deja VAE-TF", [S["dv-tf"][0][0]], [tf])
    head = ap.head_from_state(load_file(hf_hub_download("MVRL/SimLBR", "genimage/model.safetensors")))
    with torch.no_grad():
        sl = head(torch.from_numpy(C.astype(np.float32))).squeeze(1).numpy()
    report("SimLBR (released)", [np.load("runs/p4s/scores_simlbr_official_seed0.npy")], [sl])
    txt = "\n".join(lines); print(txt); (dv.W / "evasion.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
