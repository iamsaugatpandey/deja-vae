"""Figure 1 (v2): what the paper measures, and its two central findings.
(a) The instrument, on real data. The Synthbuster SDXL image at the MEDIAN fixed-point score of its class (runs/sb/bank,
    SDXL 1.0 autoencoder) and the RAISE photograph it was prompted from (geometry-matched; 68th percentile of the real
    scores) are copied by an
    autoencoder round trip R at native resolution; the copies are visually identical to the inputs. A frozen DINOv3
    reads both; the bars give how far the class token moves (cosine distance, mean over layers 8-18). The generated
    image barely moves under its own decoder (SDXL 1.0), and moves 3x more under the 0.9 version of the same
    autoencoder. Outputs: the training-free score and the two heads trained on real images only.
(b) Where: AUROC of the training-free signal per DINOv3 layer, GenImage mean over 8 generators (blue, 95% band), and
    blurred reals vs reals (grey; high = mistakes blur for generation). Cache: runs/dv/layers_full/boot.npz.
(c) Which decoder: training-free TPR at 5% FPR with the generator's exact decoder (filled) vs a near-identical
    version of it (open). Synthbuster: lossless, no re-save. GenImage: every image re-saved as JPEG q95 first (its reals are
    JPEG and its fakes PNG; runs/dv/bank_q95). Chameleon: best of seven autoencoders.
    anchor/bin/python figures/fig_overview.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent)); os.chdir(ROOT)
from style import *          # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

AS = Path("figures/fig1_assets")
LAYERS = list(range(2, 25, 2))


def arrow(fig, x0, y0, x1, y1, text=None, dy=0.0):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.8, color=INK2))
    if text:
        fig.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.3, color=INK2,
                 linespacing=1.15)


def thumb(fig, path, x, y, s, label=None):
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x, y, s, s * W / H]); ax.imshow(Image.open(path)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.5)
    if label:
        ax.set_title(label, fontsize=6.3, fontweight="normal", color=INK2, pad=2, loc="center")
    return ax


def panel_a(fig):
    W, H = fig.get_size_inches(); s = 0.105; hs = s * W / H
    yr, yf = 0.735, 0.515                      # bottom of the real / generated rows
    fig.text(0.0, 0.985, "a  Measuring the déjà vu", fontsize=8.5, fontweight="bold", va="top")
    for y, lab in ((yr, "real photograph"), (yf, "generated (SDXL)")):
        fig.text(0.0, y + hs / 2, lab, fontsize=6.6, color=INK2, va="center", rotation=90, ha="left")
    thumb(fig, AS / "real_view.png", 0.03, yr, s, "image $v$"); thumb(fig, AS / "fake_view.png", 0.03, yf, s)
    thumb(fig, AS / "real_copy_v10.png", 0.235, yr, s, "copy $R(v)$"); thumb(fig, AS / "fake_copy_v10.png", 0.235, yf, s)
    for y in (yr, yf):
        arrow(fig, 0.14, y + hs / 2, 0.23, y + hs / 2)
    fig.text(0.185, yf + hs + 0.045, "autoencoder\nround trip $R$\n(native crop)", ha="center", va="center",
             fontsize=6.2, color=INK2, linespacing=1.1)
    # frozen backbone: stack of layers, intermediate ones highlighted
    x0, w = 0.405, 0.115; yb, hb = yf, yr + hs - yf
    arrow(fig, 0.345, yr + hs / 2, x0 - 0.005, yr + hs / 2); arrow(fig, 0.345, yf + hs / 2, x0 - 0.005, yf + hs / 2)
    fig.text(0.375, yr + hs / 2 + 0.012, "$v$, $R(v)$", ha="center", va="bottom", fontsize=6.0, color=INK2)
    n = len(LAYERS)
    for i, L in enumerate(LAYERS):
        mid = 8 <= L <= 18
        fig.patches.append(Rectangle((x0, yb + i * hb / n), w, hb / n * 0.78, transform=fig.transFigure,
                                     facecolor=BLUE if mid else GRID, alpha=0.55 if mid else 1.0, lw=0))
    fig.text(x0 + w / 2, yb + hb + 0.012, "frozen DINOv3\n(24 blocks)", ha="center", va="bottom", fontsize=6.3,
             color=INK2, linespacing=1.1)
    fig.text(x0 + w + 0.006, yb + hb * (4.5 / n), "class token,\nlayers 8–18", fontsize=6.0, color=BLUE, va="center",
             linespacing=1.1)
    fig.text(x0 + w + 0.006, yb + hb * (11.2 / n), "final layer", fontsize=6.0, color=MUTED, va="center")
    # read-out: how far the class token moves
    d = np.load(AS / "layer_dist.npz"); Lm = [8, 10, 12, 14, 16, 18]
    m = {k: float(np.mean(d[k][Lm])) for k in d}
    bx, bw = 0.625, 0.10; sc = bw / 0.025
    fig.text(bx, yr + hs + 0.012, "how far it moves\n$1-\\cos(\\phi(v), \\phi(R(v)))$", fontsize=6.2, color=INK2,
             va="bottom", linespacing=1.15)
    arrow(fig, 0.583, yr + hs / 2, bx - 0.004, yr + hs / 2); arrow(fig, 0.583, yf + hs / 2, bx - 0.004, yf + hs / 2)
    def bar(y, v, col, filled=True, lab=""):
        fig.patches.append(Rectangle((bx, y), v * sc, 0.032, transform=fig.transFigure, facecolor=col if filled else "white",
                                     edgecolor=col, lw=0.8))
        fig.text(bx + v * sc + 0.006, y + 0.016, f"{v:.3f}{lab}", fontsize=6.2, color=INK2, va="center")
    bar(yr + hs / 2 - 0.016, m["real_v10"], MUTED)
    bar(yf + hs / 2 + 0.006, m["fake_v10"], BLUE, lab="  own decoder")
    bar(yf + hs / 2 - 0.046, m["fake_v09"], BLUE, filled=False, lab="  SDXL 0.9")
    # outputs
    ox = 0.855
    fig.patches.append(FancyBboxPatch((ox, yf - 0.01), 0.145, yr + hs - yf + 0.02, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor="#f4f3f0", edgecolor=GRID, lw=0.6))
    fig.text(ox + 0.008, yr + hs + 0.0, "Déjà VAE-TF", fontsize=6.6, fontweight="bold", va="top")
    fig.text(ox + 0.008, yr + hs - 0.045, "training-free:\nless movement =\nmore likely generated", fontsize=6.0,
             color=INK2, va="top", linespacing=1.1)
    fig.text(ox + 0.008, yf + 0.175, "Déjà VAE-I / -P", fontsize=6.6, fontweight="bold", va="top")
    fig.text(ox + 0.008, yf + 0.13, "head trained on real\nphotos $x$ only: $x$ (real),\n$R(x)$ (stand-in fake),\nblurred $\\tilde x$ (real)",
             fontsize=6.0, color=INK2, va="top", linespacing=1.1)


def panel_b(fig, ax):
    D = dict(np.load("runs/dv/layers_full/boot.npz"))
    for key, col, lab in (("gi|raw|dinov3/cls@{}", BLUE, "generated vs real"), ("blur|dinov3/cls@{}", MUTED, "blurred real vs real")):
        y = np.array([D[key.format(L)][0] for L in LAYERS]); lo, hi = np.array([np.percentile(D[key.format(L)][1:], [2.5, 97.5]) for L in LAYERS]).T
        ax.fill_between(LAYERS, lo, hi, color=col, alpha=0.18, lw=0); ax.plot(LAYERS, y, color=col, lw=1.3, marker="o", ms=2.4)
        if col == BLUE:
            ax.text(13, 0.86, lab, color=col, fontsize=6.4, ha="center")
        else:
            ax.text(20.5, 0.19, lab, color=col, fontsize=6.4, ha="center")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2))); ax.text(4.0, 0.47, "chance", fontsize=6, color=MUTED, va="top", ha="center")
    ax.axvspan(8, 18, color=BLUE, alpha=0.06, lw=0)
    ax.set_xlim(1, 25); ax.set_ylim(0, 1.0); ax.set_xticks([2, 8, 12, 18, 24]); ax.set_xlabel("DINOv3 layer")
    ax.set_ylabel("AUROC (GenImage)")
    ax.set_title("b  Strong mid-network, weak at the end", pad=4)


def panel_c(fig, ax):
    rows = [("Synthbuster SD 1.3, 1.4", 99.25, 23.5, "SD1 decoder", "its fine-tuned version"),
            ("Synthbuster SD 2", 95.0, 16.0, "fine-tuned decoder", "the original"),
            ("Synthbuster SDXL$^\\dagger$", 100.0, 7.0, "SDXL 1.0", "SDXL 0.9"),
            ("GenImage SD1.4, SD1.5, Wukong (JPEG q95)", 65.7, 52.0, "SD1 decoder", "its fine-tuned version")]
    for i, (lab, own, other, lo_lab, ot_lab) in enumerate(rows):
        ax.plot([other, own], [i, i], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        ax.plot([own], [i], "o", ms=5, color=BLUE, zorder=3); ax.plot([other], [i], "o", ms=5, mfc="white", mec=BLUE, mew=1.0, zorder=3)
        ax.text(own + 2.5, i, f"{own:.0f}", fontsize=6.3, va="center", color=INK)
        ax.text(other - 2.5, i, f"{other:.0f}", fontsize=6.3, va="center", ha="right", color=INK2)
        ax.text(0, i - 0.36, f"{lab}:  {lo_lab}  vs  {ot_lab}", fontsize=6.0, color=INK2, va="center")
    i = len(rows)
    ax.plot([36.3], [i], "s", ms=4.5, color=MUTED, zorder=3); ax.text(39, i, "36", fontsize=6.3, va="center", color=INK2)
    ax.text(0, i - 0.36, "Chameleon (web images): best of seven autoencoders", fontsize=6.0, color=INK2, va="center")
    ax.set_ylim(i + 0.45, -0.75); ax.set_xlim(0, 108); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("generated images caught at 5% FPR (%), no training")
    ax.plot([], [], "o", color=BLUE, ms=4.5, label="exact decoder"); ax.plot([], [], "o", mfc="white", mec=BLUE, ms=4.5, label="near-identical decoder")
    ax.legend(loc="lower right", fontsize=6.0, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.0))
    ax.set_title("c  Exact vs near-identical decoder", pad=4)


def main():
    fig = plt.figure(figsize=(TEXTW, 4.35))
    panel_a(fig)
    ax_b = fig.add_axes([0.06, 0.07, 0.33, 0.30]); panel_b(fig, ax_b)
    ax_c = fig.add_axes([0.47, 0.07, 0.52, 0.30]); panel_c(fig, ax_c)
    fig.savefig("figures/overview.pdf", bbox_inches=None); fig.savefig("figures/overview.png", dpi=220, bbox_inches=None)
    print("wrote figures/overview.pdf")


if __name__ == "__main__":
    main()
