"""Figure: where does the autoencoder fixed point live? (E1, E2; training-free)
(a) GenImage: AUROC of the training-free score -(distance between v and R(v)) per backbone layer (class token),
    mean over the 8 generators; (b) Chameleon; (c) blur control: AUROC of blur-matched real images (blurred to the
    PSNR of their own autoencoder copy) against the unblurred reals -- high = the score mistakes smoothness for
    generation; (d) E2: how the copy is made, at layer 16 and at the final layer.
Lines: DINOv3 (blue) and DINOv2 (orange) class token, 95% bootstrap bands (reals and each generator's fakes
resampled, B=1000). Grey: DINOv3 mean patch token, and PSNR(v, R(v)) as the pixel-space reference.
Data: runs/dv/layers_full/*.npz (dv_layers_full.py). Draws cached in runs/dv/layers_full/boot.npz.
    anchor/bin/python paper/figures/fig_layers.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent)); os.chdir(ROOT)
from style import *          # noqa: F401,F403

D = Path("runs/dv/layers_full"); CACHE = D / "boot.npz"
LAYERS = list(range(2, 25, 2)); BB = ["dinov3", "dinov2"]; COL = {"dinov3": BLUE, "dinov2": ORANGE}
NAME = {"dinov3": "DINOv3", "dinov2": "DINOv2"}
CONDS = ["raw", "jpeg", "x0.75", "sq"]
CLAB = {"raw": "native crop\n(Déjà VAE)", "jpeg": "native crop,\nJPEG 95 on both", "x0.75": "×0.75 resize,\nthen crop",
        "sq": "squash to 256\n(SimLBR input)"}
B = 1000


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([-pos, -neg])); n1, n0 = len(pos), len(neg)   # stored values are distances
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(n, c):
    return dict(np.load(D / f"{n}_{c}.npz"))


def keys():
    return [f"{b}/cls@{L}" for b in BB for L in LAYERS] + [f"dinov3/pmean@{L}" for L in LAYERS] + ["psnr"]


def compute():
    if CACHE.exists():
        return dict(np.load(CACHE))
    gens = sorted(f.stem[3:-4] for f in D.glob("gi_*_raw.npz") if f.stem != "gi_real_raw")
    rng = np.random.default_rng(0); res = {}
    for c in CONDS:
        real = load("gi_real", c); fk = {g: load(f"gi_{g}", c) for g in gens}
        nr, nf = len(real["x/psnr"]), {g: len(fk[g]["x/psnr"]) for g in gens}
        idx = [(np.arange(nr), {g: np.arange(nf[g]) for g in gens})] + \
              [(rng.integers(0, nr, nr), {g: rng.integers(0, nf[g], nf[g]) for g in gens}) for _ in range(B)]
        for k in keys():
            res[f"gi|{c}|{k}"] = np.array([np.mean([auroc(fk[g][f"x/{k}"][fi[g]], real[f"x/{k}"][ri]) for g in gens])
                                           for ri, fi in idx])
            if c == "raw":
                res[f"blur|{k}"] = np.array([auroc(real[f"blur/{k}"][ri], real[f"x/{k}"][ri]) for ri, _ in idx])
        print("boot", c, flush=True)
    cr, cf = load("cham_real", "raw"), load("cham_fake", "raw")
    for k in keys():
        res[f"cham|{k}"] = np.array([auroc(cf[f"x/{k}"], cr[f"x/{k}"])] +
                                    [auroc(cf[f"x/{k}"][rng.integers(0, len(cf[f'x/{k}']), len(cf[f'x/{k}']))],
                                           cr[f"x/{k}"][rng.integers(0, len(cr[f'x/{k}']), len(cr[f'x/{k}']))]) for _ in range(B)])
    np.savez(CACHE, **res)
    return res


def layer_panel(ax, res, pre, title, psnr_lab=True):
    ax.axvspan(7, 19, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
    pm = [res[f"{pre}dinov3/pmean@{L}"][0] for L in LAYERS]
    ax.plot(LAYERS, pm, color=MUTED, lw=1.0, ls=(0, (4, 1.5)), zorder=1)
    ps = res[f"{pre}psnr"][0]; ax.axhline(ps, color=INK2, lw=0.7, ls=":", zorder=1)
    for b in BB:
        M = np.array([res[f"{pre}{b}/cls@{L}"] for L in LAYERS])
        lo, hi = np.percentile(M[:, 1:], [2.5, 97.5], axis=1)
        ax.fill_between(LAYERS, lo, hi, color=COL[b], alpha=0.22, lw=0)
        ax.plot(LAYERS, M[:, 0], color=COL[b], lw=1.6, marker="o", ms=2.6, mec="white", mew=0.4, zorder=3)
    ax.set_title(title, pad=4); ax.set_xticks([2, 8, 12, 16, 20, 24]); ax.set_xlim(1, 25); ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    return pm, ps


def plot(res):
    fig = plt.figure(figsize=(TEXTW, 4.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.82], hspace=0.62, wspace=0.12)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (axb, axc):
        ax.sharey(axa); ax.tick_params(labelleft=False)
    layer_panel(axa, res, "gi|raw|", "a  GenImage")
    layer_panel(axb, res, "cham|", "b  Chameleon")
    pm, ps = layer_panel(axc, res, "blur|", "c  Blur control (lower = better)")
    axa.set_ylabel("AUROC")
    for ax in (axa, axb, axc):
        ax.set_xlabel("backbone layer")
    # direct labels (panel a carries the identities; c labels the grey references where they separate most)
    v3, v2 = res["gi|raw|dinov3/cls@10"][0], res["gi|raw|dinov2/cls@14"][0]
    axa.annotate("DINOv3", (10, v3), xytext=(9.2, 0.93), color=BLUE, fontsize=7, fontweight="bold", ha="center")
    axa.annotate("DINOv2", (24, res["gi|raw|dinov2/cls@24"][0]), xytext=(-2, -9), textcoords="offset points",
                 color=ORANGE, fontsize=7, fontweight="bold", ha="right")
    axa.text(1.6, res["gi|raw|psnr"][0] + 0.025, "PSNR", color=INK2, fontsize=6.5)
    axa.text(13, 0.06, "Déjà VAE reads 8–18", color=INK2, fontsize=6.3, ha="center")
    axa.text(1.6, 0.5 - 0.06, "chance", color=MUTED, fontsize=6.3)
    axc.text(24.4, ps - 0.075, "PSNR", color=INK2, fontsize=6.5, ha="right")
    axc.text(13.5, 0.6, "DINOv3 patch mean", color=MUTED, fontsize=6.3, ha="center")

    # (d) E2: copy conditions; layer 16 (filled) vs final layer (open)
    axd = fig.add_subplot(gs[1, :])
    xs = np.arange(len(CONDS))
    for j, b in enumerate(BB):
        for L, filled in ((16, True), (24, False)):
            for i, c in enumerate(CONDS):
                d = res[f"gi|{c}|{b}/cls@{L}"]; q = np.percentile(d[1:], [2.5, 97.5]); x = i + (j - 0.5) * 0.22
                axd.plot([x, x], q, color=COL[b], lw=0.9)
                axd.plot([x], [d[0]], "o", ms=4.2, color=COL[b] if filled else "white", mec=COL[b], mew=1.0, zorder=3)
    axd.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
    axd.set_xticks(xs); axd.set_xticklabels([CLAB[c] for c in CONDS]); axd.tick_params(axis="x", length=0)
    axd.set_xlim(-0.5, len(CONDS) - 0.5); axd.set_ylim(0.3, 0.92); axd.set_ylabel("AUROC (GenImage mean)")
    axd.set_title("d  How the copy is made (filled: layer 16; open: final layer 24)", pad=4)
    axd.grid(axis="y", color=GRID, lw=0.5); axd.set_axisbelow(True)
    axd.text(3.45, 0.5 + 0.012, "chance", color=MUTED, fontsize=6.3, ha="right")
    save(fig, "layers")


if __name__ == "__main__":
    plot(compute())
