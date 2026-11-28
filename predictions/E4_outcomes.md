# E4: outcomes

## 1. Blindness test with copies from SD1's original decoder (pre-registered: `E4_zhou_predictions.md`, commit 8ae9783; 8931a68 before the history rewrite of 2026-11-28, see README)

Run 2026-11-26 (`code/review_copies.py --ae sd1`, `--ae ftmse`; `code/zhou_compare.py`). The same 1,000 GenImage test reals,
copied once at full resolution with each autoencoder. Share flagged at the paper's 5% threshold (%), three seeds averaged;
difference with a 95% bootstrap interval over images (B = 2,000).

| Detector | Originals | ft-MSE copies | SD1-decoder copies | SD1 − ft-MSE |
|---|---|---|---|---|
| SimLBR (released; trained on SD 1.4 images) | 5.3 | 24.9 | 24.1 | −0.8 [−2.5, +1.0] |
| SimLBR (retrained, 10k SD 1.4) | 4.8 | 20.1 | 18.4 | −1.8 [−3.2, −0.3] |
| Supervised probe (SD 1.4 images) | 4.3 | 15.2 | 14.9 | −0.2 [−1.7, +1.2] |
| Reconstruction probe (copies, final layer) | 4.3 | 32.0 | 27.1 | −4.9 [−6.8, −3.1] |
| Déjà VAE-I (ft-MSE copies, layers 8–18) | 4.6 | 97.6 | 94.6 | −3.0 [−4.3, −1.8] |
| Déjà VAE-I (SD1-decoder copies, layers 8–18) | 4.5 | 89.8 | 99.6 | +9.8 [+8.2, +11.5] |
| Déjà VAE-I (ft-MSE copies, final layer) | 4.2 | 50.1 | 53.7 | +3.6 [+1.4, +5.7] |

Verdicts:
- **Z1 fails.** Probes trained on SD 1.4 images do not flag copies from SD 1.4's own decoder more often than ft-MSE copies
  (−0.8, −1.8, −0.2 points; the retrained SimLBR flags slightly fewer).
- **Z2 holds** (24.1, 18.4, 14.9%, all below 50%).
- **Z3 holds** (94.6%).
- **Z4 holds** (99.6% vs 89.8%).

Reading, by the rule written before the run: decoder mismatch does not explain the blindness. Probes trained on generated
images miss copies of real photographs even when the copy comes from the generator's own decoder, so the blindness comes
from what they were trained on. The paper's statement stands, now with the alternative explanation tested.

Not pre-specified: each Déjà VAE-I flags copies from the decoder it was trained with more often than copies from the other
one (ft-MSE-trained: +3.0 for ft-MSE copies; SD1-trained: +9.8 for SD1 copies), a trained-head counterpart of the
exact-decoder result.

Note on the run: a variable-name bug in `review_copies.py` (the output name was overwritten by a loop variable) made both
runs write to one file; the ft-MSE run's flags survived. The SD1 run was repeated with the bug fixed and reproduced the
logged numbers exactly. The analysis script and predictions were unchanged.

## 2. Synthbuster decoder bank on layers 8–18 (the trained heads' layers)

Run 2026-11-26 (`BANK_LAYERS=8,10,12,14,16,18`; `runs/sb/bank_l8`, `runs/sb/bank_crop_l8`). Same images, autoencoders,
z-scoring and in-sample threshold as the paper's bank (layers 10–18).

| Generator (own decoder) | Downscaled reals, 10–18 | Downscaled, 8–18 | Cropped reals, 10–18 | Cropped, 8–18 |
|---|---|---|---|---|
| SD 1.3 (SD1 original) | 99.5, holds | 100.0, holds | 100.0, holds | 100.0, holds |
| SD 1.4 (SD1 original) | 99.0, holds | 99.0, holds | 100.0, holds | 100.0, holds |
| SD 2 (ft-MSE) | 95.0, holds | 95.5, holds | 94.5, holds | 94.5, holds |
| SDXL (sdxl-vae, pre-specified) | 7.0, fails | 5.0, fails | 30.5, holds | 31.0, holds |
| DALL·E 3 (consistency decoder) | 34.5, holds nominally | 35.5, **fails** (FLUX 39.0) | 38.0, fails | 42.0, fails |

The SDXL version Synthbuster matches (`sdxl10`, post hoc) detects 100% under every setting.

Reading: the layer choice changes no verdict for the four Stable Diffusion versions. The only change is DALL·E 3, whose
"holds" at layers 10–18 was already described in the paper as nominal. With downscaled reals the consistency decoder,
ft-MSE and FLUX all detect 32–39% of its images, within the ±7-point binomial spread of 200 images; with cropped reals
the SDXL autoencoder detects far more (78.5% at 10–18, 86.0% at 8–18) than the consistency decoder (38.0%, 42.0%).
The paper should state that the DALL·E 3 prediction is not supported.
