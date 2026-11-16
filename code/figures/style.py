"""Shared figure style for the paper (Kay / Yang-inspired uncertainty visualisation).

Principles: distributions, not only points (half-eye slabs of bootstrap draws with a point, a thick 66% and a thin
95% interval); grey context and ONE accent per comparison; direct labels instead of legends; recessive axes, no
grid clutter; an explicit zero / chance line wherever "no effect" is a possible answer.
Palette validated with the dataviz validator (blue #2a78d6, orange #eb6834: CVD dE 24.7, normal dE 33.6).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TEXTW = 6.5            # TMLR text width (inches)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})


def halfeye(ax, y, draws, color, height=0.38, point=None, bw=None, alpha=0.35):
    """Half-eye at row y: KDE slab of `draws` above the row, point + 66% (thick) + 95% (thin) interval on it."""
    from scipy.stats import gaussian_kde
    d = np.asarray(draws, float)
    lo, hi = np.percentile(d, [0.5, 99.5]); pad = (hi - lo) * 0.15 + 1e-9
    xs = np.linspace(lo - pad, hi + pad, 200)
    if d.std() > 1e-9:
        k = gaussian_kde(d, bw_method=bw)(xs); k = k / k.max() * height
        ax.fill_between(xs, y, y + k, color=color, alpha=alpha, lw=0)
    q = np.percentile(d, [2.5, 17, 83, 97.5]); p = np.mean(d) if point is None else point
    ax.plot([q[0], q[3]], [y, y], color=color, lw=0.9, solid_capstyle="butt")
    ax.plot([q[1], q[2]], [y, y], color=color, lw=2.6, solid_capstyle="butt")
    ax.plot([p], [y], "o", ms=3.6, color=color, mec="white", mew=0.6, zorder=3)
    return q, p


def zero_line(ax, x=0, label=None, vertical=True):
    (ax.axvline if vertical else ax.axhline)(x, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=0)
    if label:
        ax.annotate(label, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=6.5, color=MUTED)


def save(fig, name):
    from pathlib import Path
    out = Path(__file__).parent
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=220)
    print("wrote", out / f"{name}.pdf")
