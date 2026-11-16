"""Figure: a matched autoencoder restores the fixed point; no autoencoder in the bank matches Chameleon (E4).
Matrix of TPR@5% of the training-free score (DINOv3 CLS layers 10-18, z-scored on the set's reals, threshold = 95th
percentile of the set's reals) per generator (rows, grouped by family) x autoencoder (columns) + the bank maximum.
Outlined: the generator's own autoencoder. Right column: the autoencoder that gives most of the generator's fakes
their strongest score (attribution), with its share. BigGAN is 128 px; the extra row thresholds it on GenImage
reals downscaled to 128 px (dv_bank_res.py). Kandinsky 2.1 and 2.2 MoVQ are bit-identical, so MoVQ appears once.
Data: runs/dv/bank_full (dv_bank_full.py, dv_bank_res.py).
    anchor/bin/python paper/figures/fig_bank.py
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent)); os.chdir(ROOT)
from style import *          # noqa: F401,F403
from matplotlib.colors import LinearSegmentedColormap

D = Path("runs/dv/bank_full")
AES = [("sd1orig", "SD1\noriginal"), ("ftmse", "SD\nft-MSE"), ("sdxl", "SDXL\n(0.9)"), ("kd21", "Kandinsky\nMoVQ"),
       ("flux", "FLUX-\nfamily"), ("dcae", "DC-AE")]
ROWS = [("gi:stable_diffusion_v_1_4", "SD1.4", "latent: SD1 AE", "sd1orig"),
        ("gi:stable_diffusion_v_1_5", "SD1.5", "", "sd1orig"),
        ("gi:wukong", "Wukong", "", "sd1orig"),
        ("gi:Midjourney", "Midjourney", "latent: AE unknown", None),
        ("gi:VQDM", "VQDM", "latent: own VQ-VAE", None),
        ("gi:ADM", "ADM", "pixel space", None),
        ("gi:glide", "GLIDE", "", None),
        ("gi:BigGAN", "BigGAN (128 px)", "GAN", None),
        ("res128", "  res.-matched reals", "", None),
        ("cham:fake", "Chameleon", "in the wild", None)]


def scores():
    A = dict(np.load(D / "phi_v.npz")); R = {k: dict(np.load(D / f"phi_R_{k}.npz")) for k, _ in AES}
    C = dict(np.load(D / "res128.npz"))

    def dist(a, b):
        return 1 - (a * b).sum(2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2) + 1e-8)

    def raw(k, g):
        return np.log(dist(C["phi_v"], C[k]) + 1e-6) if g == "real128" else np.log(dist(A[g], R[k][g]) + 1e-6)

    def z(k, g, ref):
        d, r = raw(k, g), raw(k, ref)
        return -((d - r.mean(0)) / (r.std(0) + 1e-8)).sum(1)

    T, ATT = {}, {}
    for key, *_ in ROWS:
        g, ref = ("gi:BigGAN", "real128") if key == "res128" else (key, "cham:real" if key.startswith("cham") else "gi:real")
        Sf = np.stack([z(k, g, ref) for k, _ in AES], 1); Sr = np.stack([z(k, ref, ref) for k, _ in AES], 1)
        tp = [(Sf[:, j] > np.quantile(Sr[:, j], 0.95)).mean() * 100 for j in range(len(AES))]
        mf, mr = Sf.max(1), Sr.max(1); tp.append((mf > np.quantile(mr, 0.95)).mean() * 100)
        T[key] = np.array(tp)
        w = np.bincount(Sf.argmax(1), minlength=len(AES)) / len(Sf); ATT[key] = (int(w.argmax()), w.max() * 100)
    return T, ATT


def plot(T, ATT):
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f5", BLUE, "#123f78"])
    nr, nc = len(ROWS), len(AES) + 1
    fig, ax = plt.subplots(figsize=(TEXTW, 3.35))
    for i, (key, lab, fam, match) in enumerate(ROWS):
        for j in range(nc):
            v = T[key][j]; x = j + (0.25 if j == nc - 1 else 0)
            ax.add_patch(plt.Rectangle((x, i), 0.94, 0.9, color=cmap(v / 100), lw=0))
            ax.text(x + 0.47, i + 0.47, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else INK)
            if match is not None and j < len(AES) and AES[j][0] == match:
                ax.add_patch(plt.Rectangle((x - 0.03, i - 0.03), 1.0, 0.96, fill=False, ec=ORANGE, lw=1.4))
        a, s = ATT[key]
        if key != "res128":
            pct = ">99%" if 99.5 <= s < 100 else f"{s:.0f}%"
            ax.text(nc + 0.4, i + 0.47, f"{AES[a][1].replace(chr(10), ' ')}  {pct}", va="center", fontsize=6.8, color=INK2)
        ax.text(-0.15, i + 0.47, lab, ha="right", va="center", fontsize=7.2,
                color=MUTED if key == "res128" else INK, style="italic" if key == "res128" else "normal")
        if fam:
            ax.text(-4.3, i + 0.47, fam, ha="left", va="center", fontsize=6.3, color=MUTED)
    for y in (3, 4, 5, 7, 9):
        ax.plot([-4.3, nc + 2.6], [y - 0.05, y - 0.05], color=GRID, lw=0.6, clip_on=False)
    for j, (_, name) in enumerate(AES):
        ax.text(j + 0.47, -0.25, name, ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.text(nc - 1 + 0.25 + 0.47, -0.25, "bank\nmax", ha="center", va="bottom", fontsize=6.8, color=INK2, fontweight="bold")
    ax.text(nc + 0.4, -0.25, "most often the\nstrongest (share)", ha="left", va="bottom", fontsize=6.8, color=INK2)
    ax.text(-4.3, -0.25, "generator family", ha="left", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xlim(-4.35, nc + 2.6); ax.set_ylim(nr, -1.1); ax.axis("off")
    save(fig, "bank")


if __name__ == "__main__":
    T, ATT = scores()
    for k, *_ in ROWS:
        print(f"{k:28s}", " ".join(f"{v:5.1f}" for v in T[k]), AES[ATT[k][0]][0], f"{ATT[k][1]:.1f}")
    plot(T, ATT)
