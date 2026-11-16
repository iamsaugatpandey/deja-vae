"""Data-scale curve for the trained-on-fakes baselines (fairness control for the zero-fake Déjà VAE):
SimLBR (lbr_exact, released recipe) and plain supervised (sup) trained on N reals + N SD1.4 fakes,
N in 1250/2500/5000 (runs/scale/nN, nested subsets of the P4 10k/10k set) and 10000 (runs/p4s).
Same frozen features, head, steps, thresholds. GenImage (p2, JPEG q95) and Chameleon.
    anchor/bin/python scale_report.py   ->  runs/scale/report.txt"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap

NS = [1250, 2500, 5000, 10000]
lines = ["Data-scale curve: N training reals + N training fakes (SD1.4), 3 seeds; TPR@5% / TPR@1% on calibration-set thresholds", ""]
for name, folder in (("GenImage (p2, JPEG q95)", None), ("Chameleon", "runs/cham_off")):
    out = []
    for m in ("lbr_exact", "sup"):
        for n in NS:
            if folder:
                W, tag = Path(folder), (m if n == 10000 else f"{m}~n{n}")
            else:
                W, tag = (Path("runs/p4s"), m) if n == 10000 else (Path(f"runs/scale/n{n}"), f"{m}~n{n}")
            rows = ap.read_manifest(W / "manifest.csv")
            for s in (0, 1, 2):
                f = W / f"scores_{tag}_seed{s}.npy"
                if not f.exists():
                    continue
                _, sm = ap.evaluate(np.load(f), np.load(W / f"degscores_{tag}_seed{s}.npy"), rows)
                out.append(dict(method=m, N=n, seed=s, **{k: sm[k] for k in ("mean_tpr@5%", "worst_tpr@5%", "mean_tpr@1%", "fpr_degraded_reals@5%")}))
    df = pd.DataFrame(out)
    g = df.groupby(["method", "N"])
    t = g.mean(numeric_only=True).drop(columns="seed").round(1)
    t["sd_tpr@5%"] = g["mean_tpr@5%"].std().round(1)
    lines += [f"== {name}", t.to_string(), ""]
txt = "\n".join(lines); print(txt); Path("runs/scale/report.txt").write_text(txt + "\n")
