"""Review: final-layer (v1, runs/dv, cls24) vs intermediate-layer (v2, runs/dv2, layers 8-18) heads on Synthbuster
(downscaled reals), paired, calibration and generated images resampled by RAISE photograph (sb_cluster.cluster_boot).
    anchor/bin/python review_layer_sb.py   -> runs/sb/layer.txt
"""
import os, sys
from pathlib import Path
os.environ["DV_FEAT"] = "mid"; os.environ["DV_WORK"] = "runs/dv2"
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import dejavae as dv
import sb_cluster as sc

rows, idx = dv.rows_of("sb"); ix = np.array(idx); R = [rows[i] for i in idx]
mid = dv.scores_for("sb", rows, idx)
dv.W, dv.FEAT = Path("runs/dv"), "cls24"            # same rows, final-layer work folder
fin = dv.scores_for("sb", rows, idx)
L = ["Synthbuster (downscaled reals): final layer minus layers 8-18, TPR@5% (mean over 3 seeds), photo-clustered B=1000"]
for v, name in (("dv-single", "Deja VAE-I"), ("dv", "Deja VAE-P")):
    a = [fin[v][s][0][ix] for s in (0, 1, 2)]; b = [mid[v][s][0][ix] for s in (0, 1, 2)]
    r = sc.cluster_boot(a, b, R)
    L.append(f"{name:11s} mean {r['mean'][0]:+5.1f} [{r['mean'][1]:+.1f},{r['mean'][2]:+.1f}]   "
             f"worst {r['worst'][0]:+5.1f} [{r['worst'][1]:+.1f},{r['worst'][2]:+.1f}]")
    print(L[-1], flush=True)
Path("runs/sb/layer.txt").write_text("\n".join(L) + "\n")
