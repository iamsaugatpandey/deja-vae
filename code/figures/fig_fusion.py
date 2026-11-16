"""Figure: is the autoencoder signal complementary to a semantic detector? (C7)
Forest of paired TPR@5% differences, one small multiple per benchmark (GenImage, Chameleon, size-matched Chameleon, Synthbuster).
Rows: fusion vs SimLBR alone, and fusion vs four control fusions. Half-eyes of paired bootstrap draws (B=1000; same
resample of calibration reals and test fakes for both methods and every seed; seed-averaged), exactly the procedure of
anchor_pilot.paired_bootstrap, but keeping the draws. Draws cached in runs/dv2/fusion_boot.npz.
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python paper/figures/fig_fusion.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
os.chdir(ROOT); os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from style import *          # noqa: F401,F403
import anchor_pilot as ap
import dejavae as dv

SETS = {"genimage": "GenImage", "cham": "Chameleon", "cham_m": "Size-matched\nChameleon", "sb": "Synthbuster\n(geometry-matched)"}
F = "dv+fuse-simlbr_official"
ROWS = [("simlbr_official", "vs SimLBR alone"), ("sup+fuse-simlbr_official", "vs SimLBR + supervised probe"),
        ("recon+fuse-simlbr_official", "vs SimLBR + reconstruction probe"), ("rigid-dinov2+fuse-simlbr_official", "vs SimLBR + RIGID"),
        ("dv-single+fuse-simlbr_official", "vs SimLBR + Déjà VAE-I")]
CACHE = Path("runs/dv2/fusion_boot.npz")


def runs_for(name):
    rows, idx = dv.rows_of(name); ref = Path(dv.SETS[name][1])
    runs = dv.scores_for(name, rows, idx)
    for m in ("simlbr_official", "sup", "recon", "rigid-dinov2"):
        runs[m] = {s: (np.load(ref / f"scores_{m}_seed{s}.npy"), np.load(ref / f"degscores_{m}_seed{s}.npy")) for s in (0, 1, 2)}
    ix = np.array(idx); runs = {m: {s: (a[ix], b[ix]) for s, (a, b) in v.items()} for m, v in runs.items()}
    rows = [rows[i] for i in idx]
    for v in ("dv", "dv-single", "sup", "recon", "rigid-dinov2"):
        runs[f"{v}+fuse-simlbr_official"] = {s: (dv.zfuse(runs[v][s][0], runs["simlbr_official"][s][0], rows)(runs[v][s][0], runs["simlbr_official"][s][0]), None) for s in (0, 1, 2)}
    return rows, runs


def draws(sa, sb, rows, B=1000, seed=0, eps=0.05):
    lab, split, src = ap.split_idx(rows); rng = np.random.default_rng(seed)
    cal = np.where((split == "calib") & (lab == 0))[0]; gens = sorted(set(src[(split == "test") & (lab == 1)]))
    fk = {g: np.where((split == "test") & (src == g) & (lab == 1))[0] for g in gens}
    out = np.zeros(B)
    for b in range(B):
        c = rng.choice(cal, len(cal)); fs = [rng.choice(fk[g], len(fk[g])) for g in gens]
        d = []
        for xa, xb in zip(sa, sb):
            ta, tb = ap.conformal_tau(xa[c], eps), ap.conformal_tau(xb[c], eps)
            d.append(np.mean([(xa[f] > ta).mean() for f in fs]) * 100 - np.mean([(xb[f] > tb).mean() for f in fs]) * 100)
        out[b] = np.mean(d)
    return out


def compute():
    res = dict(np.load(CACHE)) if CACHE.exists() else {}
    for name in SETS:
        if all(f"{name}|{r}" in res for r, _ in ROWS):
            continue
        rows, runs = runs_for(name)
        for r, _ in ROWS:
            res[f"{name}|{r}"] = draws([runs[F][s][0] for s in (0, 1, 2)], [runs[r][s][0] for s in (0, 1, 2)], rows)
            print(name, r, f"{res[f'{name}|{r}'].mean():+.1f}", flush=True)
        np.savez(CACHE, **res)
    return res


def plot(res):
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 2.45), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.1))
    for ax, (name, title) in zip(axes, SETS.items()):
        for k, (r, _) in enumerate(ROWS[::-1]):
            d = res[f"{name}|{r}"]
            col = BLUE if np.percentile(d, 2.5) > 0 else (ORANGE if np.percentile(d, 97.5) < 0 else MUTED)
            halfeye(ax, k, d, col)
        zero_line(ax, 0)
        ax.set_title(title, pad=5); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlim(-6, 26)
    axes[0].set_yticks(range(len(ROWS))); axes[0].set_yticklabels([lab for _, lab in ROWS[::-1]])
    axes[0].set_ylim(-0.4, len(ROWS) - 0.3)
    fig.supxlabel("Déjà VAE-P + SimLBR minus comparison: TPR at 5% FPR (points)", fontsize=8, color=INK2, y=-0.04)
    save(fig, "fusion_forest")


if __name__ == "__main__":
    plot(compute())
