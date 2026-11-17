"""Review: final-layer (v1, runs/dv, cls24) vs intermediate-layer (v2, runs/dv2, layers 8-18) heads on Synthbuster
(downscaled and cropped reals), paired, calibration and generated images resampled by RAISE photograph (sb_cluster.cluster_boot).
    anchor/bin/python review_layer_sb.py   -> runs/sb/layer.txt
"""
import os, sys
from pathlib import Path
os.environ["DV_FEAT"] = "mid"; os.environ["DV_WORK"] = "runs/dv2"
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import dejavae as dv
import sb_cluster as sc

L = ["Synthbuster: final layer minus layers 8-18, TPR@5% (mean over 3 seeds), photo-clustered B=1000"]
for SET in ("sb", "sb_crop"):
    dv.W, dv.FEAT = Path("runs/dv2"), "mid"
    rows, idx = dv.rows_of(SET); ix = np.array(idx); R = [rows[i] for i in idx]
    mid = dv.scores_for(SET, rows, idx)
    dv.W, dv.FEAT = Path("runs/dv"), "cls24"            # same rows, final-layer work folder
    fin = dv.scores_for(SET, rows, idx)
    for v, name in (("dv-single", "Deja VAE-I"), ("dv", "Deja VAE-P")):
        a = [fin[v][s][0][ix] for s in (0, 1, 2)]; b = [mid[v][s][0][ix] for s in (0, 1, 2)]
        r = sc.cluster_boot(a, b, R)
        L.append(f"[{SET}] {name:11s} mean {r['mean'][0]:+5.1f} [{r['mean'][1]:+.1f},{r['mean'][2]:+.1f}]   "
                 f"worst {r['worst'][0]:+5.1f} [{r['worst'][1]:+.1f},{r['worst'][2]:+.1f}]")
        print(L[-1], flush=True)
    ref = Path(dv.SETS[SET][1]); sl = [np.load(ref / f"scores_simlbr_official_seed{s}.npy")[ix] for s in (0, 1, 2)]
    for v, name in (("dv-single", "Deja VAE-I"), ("dv", "Deja VAE-P")):
        r = sc.cluster_boot([fin[v][s][0][ix] for s in (0, 1, 2)], sl, R)
        L.append(f"[{SET}] {name:11s} final layer - SimLBR (released): mean {r['mean'][0]:+5.1f} [{r['mean'][1]:+.1f},{r['mean'][2]:+.1f}]   "
                 f"worst {r['worst'][0]:+5.1f} [{r['worst'][1]:+.1f},{r['worst'][2]:+.1f}]")
        print(L[-1], flush=True)
Path("runs/sb/layer.txt").write_text("\n".join(L) + "\n")
