# E4: the "blindness" test with copies from SD1's original decoder (predictions, written before the run)

## Background
Zhou et al. (2026) report that final-layer probes trained on generated images are "essentially blind" to real photographs
passed through a diffusion model's autoencoder. Our version of their test (paper, appendix "What the final layer reads";
`code/review_copies.py`) copied 1,000 GenImage test reals with **ft-MSE**: probes trained on SD 1.4 images flagged 15–25% of
the copies, heads trained on copies 32–98%.

SD 1.4 decodes with **SD1's original decoder**, not ft-MSE. The earlier result therefore mixes two explanations:
- **training target:** a probe trained on generated images learns differences other than the autoencoder trace;
- **decoder mismatch:** the probe learned SD1's decoder trace, and ft-MSE leaves a different one.

Repeating the test with copies from SD1's original decoder separates them.

## Design (fixed before the run)
- Same 1,000 GenImage test reals as before (seed 5, long side at most 640 px), each copied once at full resolution, now with
  `stable-diffusion-v1-5/stable-diffusion-v1-5:vae` (SD1's original autoencoder, the bank's `sd1orig`).
- The ft-MSE run is repeated in the same code (`--ae ftmse`) so that both runs give per-image flags for the same images.
- Same detectors, pipelines and thresholds as the paper (5% FPR on the calibration reals; three seeds averaged), plus
  Déjà VAE-I trained on SD1-decoder copies (the E8 detector, `runs/dv4`).
- Analysis: `code/zhou_compare.py`, share of copies flagged and the paired difference (SD1 copies − ft-MSE copies) with a
  95% bootstrap interval over images (B = 2,000).

## Predictions
- **Z1.** The three probes trained on SD 1.4 images (SimLBR released, SimLBR retrained, supervised probe) each flag more
  SD1-decoder copies than ft-MSE copies (interval above zero).
- **Z2.** Each of the three still flags fewer than 50% of the SD1-decoder copies.
- **Z3.** Déjà VAE-I (trained on ft-MSE copies, layers 8–18) flags at least 80% of the SD1-decoder copies.
- **Z4.** Déjà VAE-I trained on SD1-decoder copies flags at least 90% of the SD1-decoder copies, and more of them than of the
  ft-MSE copies (interval above zero).

## How the outcomes will be read
- Z1 and Z2 hold: decoder mismatch accounts for part of the blindness, but most of it comes from what the probe was trained
  on. The paper's statement that the training target matters stands, with the mismatch share reported.
- Z1 fails: decoder mismatch does not explain the blindness.
- Z2 fails (a probe flags 50% or more): the blindness reported with ft-MSE copies is largely decoder mismatch. The paper's
  statement that "the training target and the layer matter" must then be weakened to the layer and the decoder.
- Every outcome is reported, including failed predictions.
