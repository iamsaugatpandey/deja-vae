# E3: exact-decoder predictions on new generators (written before any image was generated)

Date written: 2026-11-23. Committed to the repository before generation; the commit date is the record.
No image from either generator below had been generated or scored by us when this file was written.

## Generators (both Apache-2.0, run locally)
| Generator | Hugging Face id | Decoder (verified by file hash, 2026-11-23) | In the paper's bank? |
|---|---|---|---|
| Sana 600M, 512 px | `Efficient-Large-Model/Sana_600M_512px_diffusers` | DC-AE `sana-1.0` (`vae/` identical to `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`, sha256 15a4b09e…) | No. The bank holds DC-AE `in-1.0` (`mit-han-lab/dc-ae-f32c32-in-1.0`), same architecture, different weights |
| Kandinsky 2.2 | `kandinsky-community/kandinsky-2-2-prior` + `kandinsky-community/kandinsky-2-2-decoder` | MoVQ (`movq/diffusion_pytorch_model.bin` identical to Kandinsky 2.1's, sha256 772e0973…) | Yes (the "MoVQ" column); never tested on Kandinsky-generated images |

## Images
- Prompts: Synthbuster's own prompts (`prompts.csv`), one per RAISE-1k photograph, for 300 photographs drawn with seed 31 from the photographs used in the paper (the corrupt one excluded).
- One image per prompt and generator, generation seed = row index; default pipeline settings of each model card (Sana: 20 steps, guidance 4.5; Kandinsky 2.2: prior 25 steps, decoder 50 steps, guidance 4.0); 512 × 512; saved as PNG; no post-processing.
- Real side: each generated image's own RAISE photograph, matched exactly as in the paper, both ways: downscaled (centre-crop to aspect, bicubic antialiased resize) and cropped (native-resolution centre crop). Split as in the paper (seed 23).

## Bank and score (fixed now)
- Autoencoders: the paper's eight (SD1 original decoder, ft-MSE, SDXL 0.9, SDXL 1.0, MoVQ, FLUX-family, DC-AE `in-1.0`, consistency decoder), plus DC-AE `sana-1.0`, added here before generation as Sana's exact decoder.
- Score: the paper's training-free bank score (DINOv3 class-token movement, z-scored on the matched reals, layers 10–18), no JPEG re-save, threshold at the 95th percentile of the generator's own matched reals (in-sample), as in the paper's Table 1.
- Specificity: AUROC margin of the exact decoder over the best other autoencoder, re-chosen in every bootstrap draw (B = 1,000; generated and real images resampled).
- Secondary, pre-specified: the same bank on layers 8–18; and for Kandinsky, MoVQ decoding without quantisation (the Kandinsky 2.2 pipeline decodes with `force_not_quantize=True`, whereas the bank's copy quantises).

## Predictions
- **P1 (Sana, exact vs near-identical).** DC-AE `sana-1.0` is the strongest autoencoder in the Sana row, by TPR@5% and by AUROC, under both matchings, and its AUROC margin over the best other autoencoder has a 95% interval above zero.
- **P2 (Sana, near-identical version).** DC-AE `in-1.0`, the same architecture with different weights, detects at least 30 points fewer Sana images than `sana-1.0` with downscaled reals.
- **P3 (Kandinsky 2.2).** MoVQ is the strongest autoencoder in the Kandinsky row, by TPR@5% and by AUROC, under both matchings, with an AUROC-margin interval above zero.
- **P4 (level).** With downscaled reals, the exact decoder detects at least 90% of each generator's images.
- **P5 (trained detector).** Déjà VAE-I (layers 8–18, ft-MSE copies, unchanged) detects fewer images of each new generator than of SD 2 (whose decoder makes its copies) under the same matching.

## What counts as a failure
Each prediction holds or fails as stated; there is no partial credit. Every prediction is reported with its outcome, including failures. Any analysis not listed here is labelled post hoc.
