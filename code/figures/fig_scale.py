"""Appendix figure: how much generated training data do the trained-on-fakes baselines need? (fairness control)
TPR@5% of SimLBR retrained (lbr_exact) and the plain supervised probe (sup) against the number of training images
per class (nested subsets 1,250 / 2,500 / 5,000 / 10,000 real + SD1.4 fake), the released SimLBR head (162k per
class), and Déjà VAE-I (zero generated images) as a reference band. Points: mean over 3 seeds; bars: 95% bootstrap
intervals (calibration reals and each generator's test fakes resampled, B=500, seed-averaged).
Data: runs/scale/nN, runs/p4s (GenImage), runs/cham_off (Chameleon), Déjà VAE scores via dejavae.py.
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python paper/figures/fig_scale.py
"""
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
os.chdir(ROOT); os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from style import *          # noqa: F401,F403
import anchor_pilot as ap
import fig_fusion as ff

NS = [1250, 2500, 5000, 10000]; REL = 162000; B = 500
CACHE = Path("runs/scale/fig_boot.npz")
SETS = {"genimage": "GenImage", "cham": "Chameleon"}


def draws(seed_scores, rows, eps=0.05):
    lab, split, src = ap.split_idx(rows); rng = np.random.default_rng(0)
    cal = np.where((split == "calib") & (lab == 0))[0]; gens = sorted(set(src[(split == "test") & (lab == 1)]))
    fk = {g: np.where((split == "test") & (src == g) & (lab == 1))[0] for g in gens}
    out = []
    for b in range(B + 1):
        c = cal if b == 0 else rng.choice(cal, len(cal)); fs = [fk[g] if b == 0 else rng.choice(fk[g], len(fk[g])) for g in gens]
        out.append(np.mean([np.mean([(x[f] > ap.conformal_tau(x[c], eps)).mean() for f in fs]) * 100 for x in seed_scores]))
    return np.array(out)


def folder(name, m, n):
    if name == "cham":
        return Path("runs/cham_off"), (m if n == 10000 else f"{m}~n{n}")
    return (Path("runs/p4s"), m) if n == 10000 else (Path(f"runs/scale/n{n}"), f"{m}~n{n}")


def compute():
    res = dict(np.load(CACHE)) if CACHE.exists() else {}
    for name in SETS:
        for m in ("lbr_exact", "sup"):
            for n in NS:
                k = f"{name}|{m}|{n}"
                if k in res:
                    continue
                W, tag = folder(name, m, n); rows = ap.read_manifest(W / "manifest.csv")
                res[k] = draws([np.load(W / f"scores_{tag}_seed{s}.npy") for s in (0, 1, 2)], rows); print(k, f"{res[k][0]:.1f}", flush=True)
        if f"{name}|dv-single" not in res:
            rows, runs = ff.runs_for(name)
            res[f"{name}|dv-single"] = draws([runs["dv-single"][s][0] for s in (0, 1, 2)], rows)
            res[f"{name}|simlbr_official"] = draws([runs["simlbr_official"][0][0]], rows)
            print(name, "dv-single", f"{res[f'{name}|dv-single'][0]:.1f}", "released", f"{res[f'{name}|simlbr_official'][0]:.1f}", flush=True)
        np.savez(CACHE, **res)
    return res


def plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    for k, (ax, (name, title)) in enumerate(zip(axes, SETS.items())):
        d = res[f"{name}|dv-single"]; lo, hi = np.percentile(d[1:], [2.5, 97.5])
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.18, lw=0); ax.axhline(d[0], color=ORANGE, lw=1.0)
        ax.text(1100, d[0] + (1.2 if name == "genimage" else 1.2), "Déjà VAE-I (0 generated images)", color=ORANGE, fontsize=6.6, va="bottom")
        for m, col, lab in (("lbr_exact", BLUE, "SimLBR, retrained"), ("sup", INK2, "supervised probe")):
            pts = [res[f"{name}|{m}|{n}"] for n in NS]
            ax.plot(NS, [p[0] for p in pts], color=col, lw=1.4, zorder=2)
            for n, p in zip(NS, pts):
                q = np.percentile(p[1:], [2.5, 97.5]); ax.plot([n, n], q, color=col, lw=0.9)
                ax.plot([n], [p[0]], "o", ms=3.8, color=col, mec="white", mew=0.5, zorder=3)
            if name == "genimage":
                ax.text(10000 * 1.12, pts[-1][0] + (0.0 if m == "lbr_exact" else -1.6), lab, color=col, fontsize=6.6, va="center")
        r = res[f"{name}|simlbr_official"]; q = np.percentile(r[1:], [2.5, 97.5])
        ax.plot([REL, REL], q, color=BLUE, lw=0.9); ax.plot([REL], [r[0]], "D", ms=3.8, color=BLUE, mec="white", mew=0.5, zorder=3)
        ax.annotate("released\n(162k)", (REL, r[0]), xytext=(0, -16), textcoords="offset points", ha="center", va="top", fontsize=6.4, color=BLUE)
        ax.set_xscale("log"); ax.set_xlim(900, 330000)
        ax.set_xticks(NS + [REL]); ax.set_xticklabels(["1.25k", "2.5k", "5k", "10k", "162k"]); ax.minorticks_off()
        ax.set_title("ab"[k] + "  " + title, pad=4); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("generated (and real) training images per class")
    axes[0].set_ylabel("TPR at 5% FPR (%)"); axes[0].set_ylim(25, 95)
    save(fig, "scale")


if __name__ == "__main__":
    plot(compute())
