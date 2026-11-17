"""Review: is the final layer 'blind' to the autoencoder trace, or only an untrained/linear reading of it?
Zhou et al. (2026) train a LINEAR probe on final-layer pooled features to separate GenImage SD1.4 images from reals and
find it misses VAE reconstructions. Here a linear head and the paper's MLP head are trained on the SAME rows as
Deja VAE-I (real photographs = 0, their ft-MSE copies = 1, blur-matched counterfactual reals = 0; image alone),
at the final-layer class token (runs/dv) and at layers 8-18 (runs/dv2). Same optimiser, steps and seeds as dejavae train.
TPR@5% (mean over generators, threshold on calibration reals), mean over 3 seeds.
    anchor/bin/python review_linear.py   -> runs/review_linear.txt   (after run_review3.sh for the cropped-reals rows)
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch, torch.nn as nn
import anchor_pilot as ap

SETS = {"genimage": "GenImage", "cham": "Chameleon", "sb": "Synthbuster (downscaled)", "sb_crop": "Synthbuster (cropped)"}


def load(work, feat):
    os.environ["DV_WORK"], os.environ["DV_FEAT"] = work, feat
    for k in [k for k in sys.modules if k == "dejavae"]:
        del sys.modules[k]
    import dejavae as dv
    return dv


def train(X, y, linear, seed, steps=2000, batch=500, lr=1e-4, wd=1e-2, dropout=0.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    head = ap.build_head(X.shape[1], dropout, linear=linear)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]))
    head.train()
    for _ in range(steps):
        bi = rng.integers(0, len(X), batch)
        loss = lossf(head(torch.from_numpy(X[bi].astype(np.float32))).squeeze(1), torch.from_numpy(y[bi]))
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


L = [__doc__.strip().splitlines()[0], ""]
for work, feat, name in (("runs/dv", "cls24", "final layer"), ("runs/dv2", "mid", "layers 8-18")):
    dv = load(work, feat)
    rows, idx = dv.rows_of("train")
    ld = lambda k: np.load(dv.W / f"train_{k}_a.npy", mmap_mode="r")[idx]
    X = np.concatenate([ld("real"), ld("fake"), ld("cf")])
    y = np.concatenate([np.zeros(len(idx)), np.ones(len(idx)), np.zeros(len(idx))]).astype(np.float32)
    heads = {lin: [train(X, y, lin, s) for s in (0, 1, 2)] for lin in (True, False)}
    for st, sname in SETS.items():
        r, ix = dv.rows_of(st); ix = np.array(ix); R = [r[i] for i in ix]
        dp = dv.W / f"{st}_done.json"
        if not dp.exists() or len(__import__("json").load(open(dp))) < len(ix):   # features not computed in this folder
            continue
        F = np.load(dv.W / f"{st}_img_a.npy", mmap_mode="r")[ix].astype(np.float32)
        lab, split, src = ap.split_idx(R); cal = (split == "calib") & (lab == 0)
        gens = sorted(set(src[(split == "test") & (lab == 1)]))
        for lin in (True, False):
            t = []
            for h in heads[lin]:
                with torch.no_grad():
                    s = h(torch.from_numpy(F)).squeeze(1).numpy()
                tau = ap.conformal_tau(s[cal], 0.05)
                t.append(np.mean([(s[(split == "test") & (lab == 1) & (src == g)] > tau).mean() * 100 for g in gens]))
            L.append(f"{name:12s} {'linear' if lin else 'MLP':7s} {sname:26s} TPR@5 {np.mean(t):5.1f}  (seeds {', '.join(f'{v:.1f}' for v in t)})")
            print(L[-1], flush=True)
Path("runs/review_linear.txt").write_text("\n".join(L) + "\n")
