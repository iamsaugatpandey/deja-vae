"""Figure: Synthbuster under geometry matching (E5).
(a) TPR at 5% FPR per generator (rows grouped by decoder family) for SimLBR (released), Deja VAE-I, Deja VAE-P, the
    fusion Deja VAE-P + SimLBR, and RIGID (DINOv3). One pooled threshold per method/seed from the calibration reals.
(b) False-positive rate on each generator's geometry-matched TEST reals at that pooled threshold (nominal 5%):
    does matching the reals to a generator's size (e.g. shrinking a photograph to 256 px for GLIDE) itself trigger
    false alarms?
Points: mean over seeds; bars: 95% bootstrap intervals (calibration reals, the generator's fakes and its test reals
resampled, B=500). Draws cached in runs/sb/fig_boot.npz.
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python paper/figures/fig_synthbuster.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
os.chdir(ROOT); os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from style import *          # noqa: F401,F403
import anchor_pilot as ap
import fig_fusion as ff

CACHE = Path("runs/sb/fig_boot.npz"); B = 500
GENS = [("stable-diffusion-2", "SD 2", "our decoder (ft-MSE)"),
        ("stable-diffusion-1-3", "SD 1.3", "same encoder,\nother decoder"),
        ("stable-diffusion-1-4", "SD 1.4", ""),
        ("dalle3", "DALL·E 3", ""),
        ("stable-diffusion-xl", "SDXL", "other latent"),
        ("dalle2", "DALL·E 2", "pixel space"),
        ("glide", "GLIDE", ""),
        ("midjourney-v5", "Midjourney v5", "not public"),
        ("firefly", "Firefly", "")]
METHODS = [("simlbr_official", "SimLBR", INK2, "o", True), ("dv-single", "Déjà VAE-I", BLUE, "o", False),
           ("dv", "Déjà VAE-P", BLUE, "o", True), ("dv+fuse-simlbr_official", "Déjà VAE-P + SimLBR", ORANGE, "D", True),
           ("rigid-dinov3", "RIGID", MUTED, "s", True)]


def compute():
    if CACHE.exists():
        return dict(np.load(CACHE))
    rows, runs = ff.runs_for("sb")
    ref = Path("runs/sb"); _, idx = __import__("dejavae").rows_of("sb"); ix = np.array(idx)
    runs["rigid-dinov3"] = {s: (np.load(ref / f"scores_rigid-dinov3_seed{s}.npy")[ix], None) for s in (0, 1, 2)}
    lab, split, src = ap.split_idx(rows); rng = np.random.default_rng(0)
    cal = np.where((split == "calib") & (lab == 0))[0]
    fk = {g: np.where((split == "test") & (lab == 1) & (src == g))[0] for g, *_ in GENS}
    rl = {g: np.where((split == "test") & (lab == 0) & (src == g))[0] for g, *_ in GENS}
    res = {}
    for m, *_ in METHODS:
        seeds = [s for s in (0, 1, 2) if s in runs[m]]
        T = np.zeros((B + 1, len(GENS))); F = np.zeros((B + 1, len(GENS)))
        for b in range(B + 1):
            c = cal if b == 0 else rng.choice(cal, len(cal))
            for j, (g, *_) in enumerate(GENS):
                f = fk[g] if b == 0 else rng.choice(fk[g], len(fk[g])); r = rl[g] if b == 0 else rng.choice(rl[g], len(rl[g]))
                t_, f_ = [], []
                for s in seeds:
                    x = runs[m][s][0]; tau = ap.conformal_tau(x[c], 0.05)
                    t_.append((x[f] > tau).mean() * 100); f_.append((x[r] > tau).mean() * 100)
                T[b, j], F[b, j] = np.mean(t_), np.mean(f_)
        res[f"{m}|tpr"], res[f"{m}|fpr"] = T, F; print(m, np.round(T[0], 1), flush=True)
    np.savez(CACHE, **res)
    return res


def plot(res):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TEXTW, 3.3), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.06))
    n = len(METHODS); off = np.linspace(-0.3, 0.3, n)
    for ax, key in ((axa, "tpr"), (axb, "fpr")):
        for k, (m, lab, col, mk, filled) in enumerate(METHODS):
            D = res[f"{m}|{key}"]
            for j in range(len(GENS)):
                y = j + off[k]; q = np.percentile(D[1:, j], [2.5, 97.5])
                ax.plot(q, [y, y], color=col, lw=0.9)
                ax.plot([D[0, j]], [y], mk, ms=3.6, color=col if filled else "white", mec=col, mew=0.9, zorder=3)
        for y in (0.5, 3.5, 4.5, 6.5):
            ax.axhline(y, color=GRID, lw=0.6)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axb.axvline(5, color=MUTED, lw=0.7, ls=(0, (2, 2))); axb.text(5, -0.6, "nominal 5%", color=MUTED, fontsize=6.3, ha="center", va="bottom", clip_on=False)
    axa.set_xlim(0, 101); axa.set_xlabel("TPR at 5% FPR (%)"); axb.set_xlabel("FPR on matched real images (%)")
    axa.set_title("a  Generated images detected", pad=11); axb.set_title("b  False alarms on matched reals", pad=11)
    axa.set_yticks(range(len(GENS))); axa.set_yticklabels([lab for _, lab, _ in GENS]); axa.set_ylim(len(GENS) - 0.45, -0.55)
    for j, (_, _, fam) in enumerate(GENS):
        if fam:
            axa.annotate(fam, (0, j), xycoords=("axes fraction", "data"), xytext=(-62, 0), textcoords="offset points",
                         ha="right", va="center", fontsize=6.2, color=MUTED)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker=mk, ms=3.6, color=col, mfc=col if filled else "white", mec=col, mew=0.9, label=lab)
          for m, lab, col, mk, filled in METHODS]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(METHODS), frameon=False,
               fontsize=6.6, handletextpad=0.3, columnspacing=1.6)
    save(fig, "synthbuster")


if __name__ == "__main__":
    plot(compute())
