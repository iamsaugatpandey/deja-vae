"""Figure (E5): false alarms on everyday phone photographs (VISION), before and after sharing on social media.
Panels: (a) each detector's own decision rule (logit > 0); (b) the 5%-FPR threshold calibrated on camera photographs.
Rows: detectors; markers: photo version (shape + colour, so identity is never colour alone); bars: 95% intervals over
devices. Dashed line in (b): the nominal 5%.
    anchor/bin/python figures/fig_e5.py     (reads runs/e5/fpr.csv)  -> e5.pdf/.png
"""
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent))
from style import *          # noqa: F401,F403
import matplotlib.pyplot as plt

VERS = [("nat", "as taken", INK, "o", True), ("natFBH", "Facebook, high quality", BLUE, "^", True),
        ("natFBL", "Facebook, low quality", BLUE, "^", False), ("natWA", "WhatsApp", ORANGE, "s", True),
        ("flat", "low texture (walls, skies)", MUTED, "D", False)]
SHORT = {"SimLBR (released)": "SimLBR", "Deja VAE-I": "Déjà VAE-I"}


def main():
    D = {}
    for r in csv.DictReader(open(ROOT / "runs/e5/fpr.csv")):
        D[(r["threshold"], r["detector"], r["version"])] = (float(r["fpr"]), float(r["lo"]), float(r["hi"]))
    panels = [("own", "a  Each detector's own decision rule"), ("camera", "b  Threshold set at 5% FPR on camera photos")]
    dets = sorted({d for t, d, v in D if t == "camera"}, key=lambda d: -D[("camera", d, "nat")][0])
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 0.36 * len(dets) + 1.1), sharey=True, gridspec_kw=dict(wspace=0.08))
    off = np.linspace(-0.28, 0.28, len(VERS))
    for ax, (t, title) in zip(axes, panels):
        for i, d in enumerate(dets):
            if t == "own" and not any((t, d, v) in D for v, *_ in VERS):
                ax.text(1, i, "no default decision rule", va="center", fontsize=6.5, color=MUTED, style="italic")
            for k, (v, lab, c, m, filled) in enumerate(VERS):
                if (t, d, v) not in D:
                    continue
                p, lo, hi = D[(t, d, v)]; y = i + off[k]
                ax.plot([lo, hi], [y, y], color=c, lw=0.8, alpha=0.9, clip_on=False)
                ax.plot([p], [y], m, ms=4.2, color=c, mfc=c if filled else "white", mec=c, mew=0.9, zorder=3,
                        clip_on=False, label=lab if (i == 0 and ax is axes[0]) else None)
        if t == "camera":
            zero_line(ax, 5, "nominal 5%")
        ax.set_title(title, fontsize=7.5); ax.set_xlabel("Real photos flagged as generated (%)")
        # square-root scale: most false-alarm rates lie between 0 and 10%, where a linear axis stacks them on the spine
        ax.set_xscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.square(x)))
        ax.set_xlim(0, 100); ax.set_xticks([0, 1, 5, 10, 20, 40, 60, 80, 100]); ax.set_xticklabels(["0", "1", "5", "10", "20", "40", "60", "80", "100"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.spines["left"].set_position(("outward", 6)); ax.spines["bottom"].set_position(("outward", 3))
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(dets))); axes[0].set_yticklabels([SHORT.get(d, d) for d in dets]); axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    fig.legend(loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.075), handletextpad=0.3, columnspacing=1.2)
    save(fig, "e5")


if __name__ == "__main__":
    main()
