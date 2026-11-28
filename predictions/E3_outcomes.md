# E3: outcomes of the pre-registered predictions

Predictions: `E3_predictions.md` (commit 41be8c4, 2026-11-23 12:16 ET, before any image was generated; 4ecfdaf before the history rewrite of 2026-11-28, see README).
Images generated 2026-11-23 (`e3_generate.py`); bank and detector scored 2026-11-24 (`e3_bank.py run`, `e3_bank.py detector`).
Full output: `runs/e3/bank/report.txt`, `runs/e3/detector.txt`.

## Results (primary analysis: layers 10–18; 300 generated images and 300 matched reals per generator)

TPR (%) at the in-sample 5% threshold, downscaled reals:

| Generator | SD1 | ft-MSE | SDXL 0.9 | MoVQ | FLUX | DC-AE in-1.0 | Cons. | SDXL 1.0 | **DC-AE sana-1.0** |
|---|---|---|---|---|---|---|---|---|---|
| Sana 600M | 53.3 | 22.3 | 8.3 | 14.3 | 8.7 | 19.7 | 1.0 | 17.7 | **97.7** |
| Kandinsky 2.2 | 77.3 | 53.7 | 14.0 | **99.0** | 44.3 | 19.3 | 38.0 | 5.7 | 60.7 |

AUROC margin of the exact decoder over the best other autoencoder (re-chosen in every draw; 95% interval):

| | Downscaled reals | Cropped reals |
|---|---|---|
| Sana (DC-AE sana-1.0) | +0.104 [+0.081, +0.128] | +0.037 [+0.025, +0.050] |
| Kandinsky 2.2 (MoVQ) | +0.057 [+0.040, +0.074] | +0.024 [+0.015, +0.035] |

## Verdicts
- **P1 holds.** DC-AE sana-1.0 is the row maximum for Sana by TPR and AUROC under both matchings; margin interval above zero.
- **P2 holds.** DC-AE in-1.0, the same architecture with different weights, detects 19.7% against 97.7% (78.0 points fewer).
- **P3 holds.** MoVQ is the row maximum for Kandinsky 2.2 under both matchings; margin interval above zero.
- **P4 holds.** Exact decoder with downscaled reals: Sana 97.7%, Kandinsky 99.0%.
- **P5 fails.** Déjà VAE-I detects Sana 98.4% and Kandinsky 98.8% with downscaled reals (SD 2: 98.8%), and with cropped reals Kandinsky 94.8% against SD 2's 93.7%. The trained detector does not detect the new generators less well than SD 2; prediction P5 was wrong.

## Secondary analyses (pre-specified)
- Layers 8–18 give the same verdicts (Sana margins +0.111 and +0.031, Kandinsky +0.060 and +0.022; all intervals above zero).
- MoVQ decoded without quantisation (as the Kandinsky 2.2 pipeline decodes) detects 98.3% (downscaled) and 100% (cropped), essentially the same as with quantisation.

## Not pre-specified (reported as post hoc)
- SD1's original decoder detects 77.3% of Kandinsky 2.2 images (AUROC 0.942), the second-highest in that row; we have no explanation.
