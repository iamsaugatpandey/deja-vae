"""E4: Zhou blindness test, copies from SD1's ORIGINAL decoder vs copies from ft-MSE (paired, the same 1,000 GenImage
test reals). Predictions Z1-Z4 and decision rules: paper/deja-vae/predictions/E4_zhou_predictions.md (written before the run).
Inputs: runs/review_copies_{ftmse,sd1}_flags.npz (review_copies.py --ae ...). Per detector: share of copies flagged
(seed-averaged), difference SD1 - ft-MSE with a 95% bootstrap interval over images (B = 2,000, seed 0).
    anchor/bin/python zhou_compare.py   -> runs/zhou_compare.txt
"""
from pathlib import Path
import numpy as np

PROBES = ["SimLBR (released; trained on SD1.4 images)", "SimLBR (retrained, 10k SD1.4)", "Supervised probe (SD1.4 images)"]


def main():
    A, B = dict(np.load("runs/review_copies_ftmse_flags.npz")), dict(np.load("runs/review_copies_sd1_flags.npz"))
    assert (A["pick"] == B["pick"]).all(), "the two runs must copy the same images"
    names = [k[:-5] for k in A if k.endswith("|copy")]
    rng = np.random.default_rng(0); n = len(A["pick"]); boot = rng.integers(0, n, (2000, n))
    L = [__doc__.strip().splitlines()[0], f"n = {n} images; share flagged (%), seed-averaged", "",
         f"{'detector':48s} {'orig(ft)':>8s} {'orig(sd1)':>9s} {'ft-MSE':>7s} {'SD1':>6s}   SD1 - ft-MSE [95% CI]"]
    R = {}
    for k in names:
        a, b = A[f"{k}|copy"].mean(0) * 100, B[f"{k}|copy"].mean(0) * 100      # per image, seed-averaged
        d = b - a; ci = np.percentile(d[boot].mean(1), [2.5, 97.5])
        R[k] = (a.mean(), b.mean(), d.mean(), ci)
        L.append(f"{k:48s} {A[k + '|orig'].mean() * 100:8.1f} {B[k + '|orig'].mean() * 100:9.1f} {a.mean():7.1f} {b.mean():6.1f}"
                 f"   {d.mean():+6.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")
    L += ["", "Pre-specified checks:"]
    z1 = all(R[k][3][0] > 0 for k in PROBES)
    L.append(f"  Z1 SD1.4-trained probes flag more SD1-decoder copies than ft-MSE copies (all three CIs above 0): {'HOLDS' if z1 else 'FAILS'}")
    z2 = all(R[k][1] < 50 for k in PROBES)
    L.append(f"  Z2 all three flag fewer than 50% of SD1-decoder copies: {'HOLDS' if z2 else 'FAILS'}  "
             f"({', '.join(f'{R[k][1]:.1f}' for k in PROBES)})")
    k3 = "Deja VAE-I (copies, layers 8-18)"
    L.append(f"  Z3 Deja VAE-I (ft-MSE copies, layers 8-18) flags at least 80% of SD1-decoder copies: {'HOLDS' if R[k3][1] >= 80 else 'FAILS'} ({R[k3][1]:.1f})")
    k4 = "Deja VAE-I, SD1-decoder copies (layers 8-18)"
    z4 = R[k4][1] >= 90 and R[k4][3][0] > 0
    L.append(f"  Z4 Deja VAE-I trained on SD1-decoder copies flags at least 90% of SD1-decoder copies and more of them than of "
             f"ft-MSE copies: {'HOLDS' if z4 else 'FAILS'} ({R[k4][1]:.1f} vs {R[k4][0]:.1f})")
    txt = "\n".join(L); print(txt); Path("runs/zhou_compare.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
