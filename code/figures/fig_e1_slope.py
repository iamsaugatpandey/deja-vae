"""Figure (E1): released detectors and the paper's methods under original vs matched benchmark versions (slope chart).
One panel per benchmark; one line per detector; points = TPR at 5% FPR (mean over generators), vertical bars = 95%
bootstrap intervals (e1_stats.py: Synthbuster resampled by RAISE photograph, others by image; B = 1,000).
Grey context, two accents per panel where the panel has a story; direct labels instead of a legend.
    anchor/bin/python paper/figures/fig_e1_slope.py        (reads runs/e1/stats_tpr.csv)  -> e1_slope.pdf/.png
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).parent))
from style import *          # noqa: F401,F403
import matplotlib.pyplot as plt

SHORT = {"SimLBR (released)": "SimLBR", "Deja VAE-I": "Déjà VAE-I", "AlignedForensics": "AlignedForensics"}
PANELS = [  # (benchmark, versions, x labels, title, accents {detector: colour})
    ("GenImage", ["genimage_orig", "genimage_q95"], ["as\ndistributed", "JPEG re-save\n(format cue gone)"],
     "a  GenImage: format cue", {"NPR": ORANGE}),
    ("Chameleon", ["cham", "cham_c", "cham_m"], ["as\ndistributed", "resize +\nJPEG", "size-\nmatched"],
     "b  Chameleon: processing vs width", {"B-Free": ORANGE, "SimLBR (released)": BLUE}),
    ("Synthbuster", ["sb_q95", "sb_crop_q95"], ["reals\ndownscaled", "reals\ncropped"],
     "c  Synthbuster: how reals are matched", {"Deja VAE-I": BLUE, "SimLBR (released)": ORANGE}),
    ("EvalGEN", ["eg_q95", "eg_crop_q95"], ["reals\ndownscaled", "reals\ncropped"],
     "d  EvalGEN (2024–25 generators)", {}),
]


def repel(ys, gap):
    """Shift label positions apart (keeps order) so that neighbours are at least `gap` apart."""
    o = np.argsort(ys); y = np.array(ys, float)[o]
    for _ in range(200):
        moved = False
        for k in range(1, len(y)):
            if y[k] - y[k - 1] < gap:
                d = (gap - (y[k] - y[k - 1])) / 2; y[k - 1] -= d; y[k] += d; moved = True
        if not moved:
            break
    out = np.empty_like(y); out[o] = y
    return out


def main():
    T = {}
    for r in csv.DictReader(open(ROOT / "runs/e1/stats_tpr.csv")):
        T[(r["version"], r["detector"])] = (float(r["tpr5"]), float(r["lo"]), float(r["hi"]))
    panels = [p for p in PANELS if any((p[1][0], d) in T for _, d in T)]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXTW, 2.7), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(p[1]) for p in panels], wspace=0.95))
    axes = np.atleast_1d(axes)
    for ax, (bench, vers, xl, title, acc) in zip(axes, panels):
        dets = sorted({d for v, d in T if v in vers}, key=lambda d: d in acc)      # accents drawn last (on top)
        ends = []
        for d in dets:
            xs = [k for k, v in enumerate(vers) if (v, d) in T]
            if len(xs) < 2:
                continue
            m = [T[(vers[k], d)] for k in xs]; c = acc.get(d, MUTED); lw = 1.6 if d in acc else 0.8
            ax.plot(xs, [v[0] for v in m], color=c, lw=lw, zorder=3 if d in acc else 2, solid_capstyle="round")
            for x, (p, lo, hi) in zip(xs, m):
                ax.plot([x, x], [lo, hi], color=c, lw=0.7, alpha=0.9, zorder=2)
                ax.plot([x], [p], "o", ms=3.2 if d in acc else 2.4, color=c, mec="white", mew=0.5, zorder=4)
            ends.append((d, m[-1][0], c, d in acc))
        ly = repel([e[1] for e in ends], 6.2)
        for (d, y, c, a), yy in zip(ends, ly):
            ax.annotate(SHORT.get(d, d), (len(vers) - 1, y), xytext=(len(vers) - 1 + 0.12, yy), textcoords="data",
                        fontsize=6.3, color=INK if a else INK2, fontweight="bold" if a else "normal", va="center", ha="left",
                        annotation_clip=False)
        ax.set_xticks(range(len(vers))); ax.set_xticklabels(xl, fontsize=6.3)
        ax.set_xlim(-0.25, len(vers) - 1 + 0.1); ax.set_ylim(0, 100); ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        for yv in (25, 50, 75):
            ax.axhline(yv, color=GRID, lw=0.5, zorder=0)
    axes[0].set_ylabel("Generated images detected (%)\nat 5% false-positive rate")
    save(fig, "e1_slope")


if __name__ == "__main__":
    main()
