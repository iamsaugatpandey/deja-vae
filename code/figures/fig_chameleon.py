"""Figures for the Chameleon audit (C6).
  chameleon_audit.pdf   (a) one dot per image: width x height, real vs generated; (b) metadata-only AUROC, original vs size-matched
  chameleon_decomp.pdf  processing cost vs shortcut cost per detector (paired bootstrap half-eyes, zero lines)
Run from the repo root:  anchor/bin/python figures/fig_chameleon.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import os; os.chdir(ROOT)
from style import *          # noqa: F401,F403
import anchor_pilot as ap

NAMES = {"simlbr_official": "SimLBR (released)", "lbr_exact": "SimLBR (retrained)", "sup": "Supervised probe",
         "recon": "Reconstruction probe", "dv": "Déjà VAE-P", "dv-single": "Déjà VAE-I",
         "dv-tf": "Déjà VAE-TF", "rigid-dinov3": "RIGID (DINOv3)", "dv+fuse-simlbr_official": "Déjà VAE-P + SimLBR"}


def sizes():
    f = Path("runs/cham_m/sizes_original.npz")
    if not f.exists():
        from PIL import Image
        rows = ap.read_manifest(Path("runs/cham/manifest.csv"))
        wh = []
        for r in rows:
            with Image.open(r["path"]) as im:
                wh.append(im.size)
        np.savez(f, wh=np.array(wh), y=np.array([r["label"] for r in rows]))
    z = np.load(f); return z["wh"], z["y"]


def audit_table():
    lines = Path("runs/cham_m/audit.txt").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.split()[:2] == ["original", "matched"])
    out = {}
    for l in lines[i + 1:]:
        p = l.split()
        if len(p) < 3:
            break
        out[" ".join(p[:-2])] = (float(p[-2]), float(p[-1]))
    return out


def fig_audit():
    wh, y = sizes(); A = audit_table()
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
    rng = np.random.default_rng(0); jit = lambda n: np.exp(rng.normal(0, 0.006, n))   # tiny jitter: dots stay countable
    for lab, c, z in ((1, ORANGE, 1), (0, BLUE, 2)):
        m = y == lab
        a.scatter(wh[m, 0] * jit(m.sum()), wh[m, 1] * jit(m.sum()), s=1.2, color=c, alpha=0.22, lw=0, zorder=z, rasterized=True)
    a.set_xscale("log"); a.set_yscale("log")
    ticks = [250, 500, 1000, 2000, 4000]
    a.set_xticks(ticks); a.set_xticklabels(ticks); a.set_yticks(ticks); a.set_yticklabels(ticks)
    a.minorticks_off(); a.set_xlim(250, 6000); a.set_ylim(80, 7000)
    a.set_xlabel("width (px)"); a.set_ylabel("height (px)")
    a.annotate(f"real: all {int((y == 0).sum()):,}\nare 500 px wide", (510, 130), xytext=(640, 105), fontsize=7, color=BLUE,
               ha="left", va="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
    a.annotate(f"generated: {int((y == 1).sum()):,} images,\n{int((wh[y == 1, 0] == 500).sum())} at 500 px width",
               (1500, 3000), fontsize=7, color=ORANGE, ha="left", va="bottom")
    a.set_title("a   One dot per Chameleon image", pad=6)

    order = ["rule width!=500", "width", "width+height", "aspect", "format+quant-table", "bytes/pixel", "all six"]
    label = {"rule width!=500": "rule: width ≠ 500", "width": "width", "width+height": "width + height", "aspect": "aspect ratio",
             "format+quant-table": "format + JPEG table", "bytes/pixel": "bytes per pixel", "all six": "all six features"}
    for k, key in enumerate(order[::-1]):
        o, m = A[key]
        b.plot([m, o], [k, k], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        b.plot([o], [k], "o", ms=4.5, color=INK2, zorder=3); b.plot([m], [k], "o", ms=4.5, color=BLUE, zorder=3)
    b.set_yticks(range(len(order))); b.set_yticklabels([label[k] for k in order[::-1]])
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
    b.annotate("chance", (0.5, len(order) - 0.45), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=MUTED)
    b.set_xlim(0.45, 1.03); b.set_xlabel("AUROC from metadata alone (5-fold CV)")
    b.set_ylim(-1.1, len(order) - 0.25)
    b.annotate("original", (A["all six"][0], 0), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7, color=INK2)
    b.annotate("size-matched", (A["all six"][1], 0), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7, color=BLUE)
    b.set_title("b   What metadata alone reveals", pad=6)
    save(fig, "chameleon_audit")


def fig_decomp():
    Z = np.load("runs/cham_m/decomp_boot.npz")
    methods = [m for m in NAMES if f"{m}|total" in Z.files]
    methods.sort(key=lambda m: Z[f"{m}|total"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, col, title in ((axes[0], "processing", ORANGE, "a   Cost of one extra resize + JPEG"),
                                (axes[1], "geometry", BLUE, "b   Cost of removing the size shortcut")):
        for k, m in enumerate(methods):
            halfeye(ax, k, -Z[f"{m}|{key}"], col)   # plotted as change in TPR (negative = detector loses points)
        zero_line(ax, 0, "no change" if key == "geometry" else None)
        ax.set_xlim(-13.5, 4.5); ax.set_xlabel("change in TPR at 5% FPR (points)")
        ax.set_title(title, pad=6); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(methods))); axes[0].set_yticklabels([NAMES[m] for m in methods])
    axes[0].set_ylim(-0.4, len(methods) - 0.35)
    save(fig, "chameleon_decomp")


if __name__ == "__main__":
    fig_audit(); fig_decomp()
