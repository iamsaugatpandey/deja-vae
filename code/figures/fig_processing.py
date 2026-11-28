"""Figure: one resize or re-compression moves generated images off the fixed point (C5).
(a) TPR@5% with the threshold re-set on equally processed reals; (b) FPR on processed reals at the UNprocessed threshold.
Methods: Déjà VAE-P (v2), Déjà VAE-P with processing-aware training (v3), SimLBR (released).
Point = mean over 3 generators (SD1.4, Wukong, Midjourney) and 3 seeds; thick/thin bars = 66%/95% bootstrap intervals
(reals and each generator's fakes resampled, B=1000). Data: runs/dv{2,3}/resize/post_*.npz.
    anchor/bin/python figures/fig_processing.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent)); os.chdir(ROOT)
from style import *          # noqa: F401,F403

CONDS = ["none", "x0.75", "x0.5", "x1.5", "q75", "x0.75+q75"]
LABEL = {"none": "none", "x0.75": "×0.75", "x0.5": "×0.5", "x1.5": "×1.5", "q75": "JPEG 75", "x0.75+q75": "×0.75 +\nJPEG 75"}
GENS = ["stable_diffusion_v_1_4", "wukong", "Midjourney"]
METHODS = [("runs/dv2", "dv", (0, 1, 2), BLUE, "Déjà VAE-P"), ("runs/dv3", "dv", (0, 1, 2), ORANGE, "Déjà VAE-P, processing-aware training"),
           ("runs/dv2", "simlbr_official", (0,), INK2, "SimLBR (released)")]
B = 1000


def stats(folder, m, seeds):
    D = {c: dict(np.load(f"{folder}/resize/post_{c}.npz")) for c in CONDS}
    rng = np.random.default_rng(0); tpr = {c: [] for c in CONDS}; fpr = {c: [] for c in CONDS}
    nr = len(D["none"][f"real|{m}/{seeds[0]}"])
    for b in range(B + 1):                     # b = 0: point estimate on the full sample
        ri = np.arange(nr) if b == 0 else rng.integers(0, nr, nr)
        fi = {g: (np.arange(len(D["none"][f"{g}|{m}/{seeds[0]}"])) if b == 0 else rng.integers(0, len(D["none"][f"{g}|{m}/{seeds[0]}"]), len(D["none"][f"{g}|{m}/{seeds[0]}"]))) for g in GENS}
        for c in CONDS:
            t_, f_ = [], []
            for s in seeds:
                real = D[c][f"real|{m}/{s}"][ri]; thr = np.quantile(real, 0.95)
                t_.append(np.mean([(D[c][f"{g}|{m}/{s}"][fi[g]] > thr).mean() for g in GENS]) * 100)
                f_.append((real > np.quantile(D["none"][f"real|{m}/{s}"][ri], 0.95)).mean() * 100)
            tpr[c].append(np.mean(t_)); fpr[c].append(np.mean(f_))
    return {c: np.array(v) for c, v in tpr.items()}, {c: np.array(v) for c, v in fpr.items()}


def pointrange(ax, x, d, col):
    q = np.percentile(d[1:], [2.5, 17, 83, 97.5])
    ax.plot([x, x], [q[0], q[3]], color=col, lw=0.9); ax.plot([x, x], [q[1], q[2]], color=col, lw=2.6)
    ax.plot([x], [d[0]], "o", ms=3.8, color=col, mec="white", mew=0.6, zorder=3)


def main():
    S = [(stats(f, m, s), col, lab) for f, m, s, col, lab in METHODS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTW, 2.5), gridspec_kw=dict(wspace=0.28))
    off = np.linspace(-0.22, 0.22, len(S))
    for k, ((T, Fp), col, lab) in enumerate(S):
        for i, c in enumerate(CONDS):
            pointrange(a, i + off[k], T[c], col); pointrange(b, i + off[k], Fp[c], col)
    for ax in (a, b):
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([LABEL[c] for c in CONDS], fontsize=6.5)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-0.5, len(CONDS) - 0.5)
        ax.set_xlabel("processing applied after generation")
    a.set_ylabel("TPR at 5% FPR (%)"); a.set_ylim(50, 101)
    a.set_title("a   Detection after processing", pad=6)
    b.axhline(5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
    b.annotate("nominal 5%", (-0.5, 5), xytext=(1, -5), textcoords="offset points", ha="left", va="top", fontsize=6.5, color=MUTED)
    b.set_ylabel("FPR on processed reals (%)"); b.set_ylim(0, 31)
    b.set_title("b   False alarms if the threshold is not re-set", pad=6)
    # direct labels (no legend box): name each series once, next to its 'none' point in panel a
    for k, (_, col, lab) in enumerate(S):         # direct, coloured series names in empty space (no legend box)
        a.text(0.03, 0.06 + 0.075 * (len(S) - 1 - k), lab, transform=a.transAxes, color=col, fontsize=7, ha="left", va="bottom")
    save(fig, "processing")


if __name__ == "__main__":
    main()
