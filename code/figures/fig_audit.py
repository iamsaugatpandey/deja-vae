"""Figure (E2): metadata-only audit heat map. Rows: benchmark versions (as distributed, then our matched versions);
columns: feature groups; cell = 5-fold CV AUROC of a classifier that sees no pixels (0.5 = no information).
Sequential single hue (blue), light -> dark from 0.5 to 1.0; values printed in each cell (text ink, not series colour).
    anchor/bin/python figures/fig_audit.py     (reads runs/audit/audit.txt)  -> audit.pdf/.png
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(Path(__file__).parent))
from style import *          # noqa: F401,F403
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROWS = [("genimage", "GenImage"), ("chameleon", "Chameleon"), ("synthbuster_as_distributed", "Synthbuster"),
        ("wildrf_test", "WildRF (social media)"), ("commfor_eval_sample", "Community Forensics (sample)"),
        ("robust_ldm_original", "Robust LDM"), ("robust_ldm_postprocessed", "Robust LDM, post-processed"),
        None,
        ("genimage_size_matched", "GenImage, size-matched"), ("chameleon_size_matched", "Chameleon, size-matched"),
        ("synthbuster_downscaled", "Synthbuster, downscaled reals"), ("synthbuster_cropped", "Synthbuster, cropped reals"),
        ("evalgen_downscaled", "EvalGEN + RAISE, downscaled"), ("evalgen_cropped", "EvalGEN + RAISE, cropped")]
COLS = [("geometry", "Size"), ("format", "Format"), ("provenance", "EXIF / ICC"), ("bytes/pixel", "Bytes per\npixel"),
        ("all", "All")]


def main():
    lines = (ROOT / "runs/audit/audit.txt").read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith("geometry"))
    names = lines[hdr].split(); V = {}
    for l in lines[hdr + 1:]:
        p = l.split()
        if len(p) == len(names) + 1:
            V[p[0]] = dict(zip(names, map(float, p[1:])))
    rows = [r for r in ROWS if r is None or r[0] in V]
    M = np.array([[V[r[0]][c] if r else np.nan for c, _ in COLS] for r in rows])
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#b9d3f3", "#6fa3e6", BLUE, "#123f7a"])
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.imshow(np.clip((M - 0.5) / 0.5, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j in range(len(COLS)):
            if r is None:
                continue
            v = M[i, j]; dark = (v - 0.5) / 0.5 > 0.55
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6, color="white" if dark else INK,
                    fontweight="bold" if j == len(COLS) - 1 else "normal")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c for _, c in COLS], fontsize=6.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([i for i, r in enumerate(rows) if r]); ax.set_yticklabels([r[1] for r in rows if r], fontsize=6.8)
    k = rows.index(None)
    ax.axhspan(k - 0.5, k + 0.5, color="white", zorder=2)
    ax.text(-0.6, k, "our matched versions", ha="right", va="center", fontsize=6.6, color=MUTED, style="italic",
            transform=ax.transData)
    ax.text(-0.6, -1.15, "as distributed", ha="right", va="center", fontsize=6.6, color=MUTED, style="italic")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(len(COLS) - 1.5, color="white", lw=2.5)
    ax.set_title("AUROC of a classifier that sees no pixels (0.5 = no information)", fontsize=7, pad=22, loc="left",
                 fontweight="normal", color=INK2)
    save(fig, "audit")


if __name__ == "__main__":
    main()
