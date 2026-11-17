# Déjà VAE: When Do Autoencoder Traces Reveal Generated Images?

Code, per-image scores and benchmark-matching lists for the paper of the same title (under review at TMLR).

A latent diffusion model's output has already passed through the model's autoencoder, so encoding and decoding it again
changes it very little. This repository measures when that "déjà vu" reveals a generated image: where it is visible inside a
frozen foundation model, which decoder it belongs to, and what removes it. The paper also includes a size-matched version of
Chameleon and two size-matched evaluations of Synthbuster (real photographs downscaled or cropped to each generated image's size).

The repository contains **no dataset images**, apart from the two 256-pixel crops shown in Figure 1 and their autoencoder copies (`code/figures/fig1_assets/`). The datasets are available from their authors (see "Data").

## Quick check (one minute, no GPU, no images)

```bash
cd code
pip install numpy pandas
python reproduce_main_table.py ../data/scores
```

This recomputes every cell of Table 1 (TPR at 5% and 1% FPR, worst generator, and false-positive rate on degraded reals,
for GenImage, Chameleon and Synthbuster) and of Table 2 (Synthbuster with downscaled and with cropped reals) from the
released per-image scores.

## Contents

```
README.md
LICENSE                       MIT
data/
  scores/<benchmark>.csv.gz   per-image scores of every method and seed (see "Score files" below)
  scores/aeroblade_*.csv.gz   AEROBLADE LPIPS distances on the fixed stratified subsets
  matching/                   lists that rebuild size-matched Chameleon, its control and the Synthbuster split
  decoder_bank_synthbuster_prespecified.txt   Table 3 (top) as produced before the post-hoc SDXL analysis
  decoder_bank_synthbuster_with_posthoc.txt   the same report including the post-hoc SDXL 1.0 column
  decoder_bank_synthbuster_cropped.txt        Table 3 (bottom): the bank with cropped reals
code/
  requirements.txt
  reproduce_main_table.py     Tables 1 and 2 from data/scores
  *.py                        experiments (see "Paper to code")
  pipelines/*.sh              the order in which the experiments were run
  figures/*.py                every figure in the paper
```

## Score files

One row per evaluation image (calibration and test splits). Columns:

- `image`: path relative to the dataset's root, as distributed by the dataset's authors. Prefixes:
  - `genimage/`
  - `chameleon_data/`
  - `synthbuster/synthbuster/` for Synthbuster's generated images
  - `synthbuster_real_matched/<generator>/<raise_id>.png` for RAISE-1k photographs downscaled to that generator's geometry
  - `synthbuster_real_cropped/<generator>/<raise_id>.png` for native-resolution crops of the same photographs (no resampling)
  - `cham_matched/` and `cham_ctrl/` for our transformed Chameleon fakes
- `split`: `calib` or `test`.
- `label`: 1 for generated, 0 for real.
- `source`: the generator.
- `<method>@seed<k>`: the score; higher means more likely generated. `<method>@seed<k>|degraded` is the score of the
  degraded copy of the same image, which is only used for test reals.

Deterministic methods have a single seed column (RIGID, Déjà VAE-TF). The released SimLBR head is one fixed model, so its
three seed columns are identical. Methods ending in `_SD1decoder` are the retrained detectors of Section 6 (Table 7);
methods ending in `_finallayer` read the final-layer class token instead of layers 8-18 (App. E, Table 5).
`synthbuster_crop_matched.csv.gz` has the same generated images as `synthbuster_geometry_matched.csv.gz`, with cropped reals.
`aeroblade_synthbuster_no_resave.csv.gz` scores the same subset rows as stored, without the JPEG q95 re-save.
Fusions are not stored; they are the sum of two scores, each standardised on the calibration reals
(`reproduce_main_table.py` shows how).

| Column prefix | Paper name |
|---|---|
| `SimLBR_released`, `SimLBR_retrained` | SimLBR (released / retrained) |
| `supervised_probe`, `reconstruction_probe` | Supervised probe, Reconstruction probe |
| `RIGID_dinov2_pixel`, `RIGID_dinov2_norm`, `RIGID_dinov3` | RIGID variants |
| `DejaVAE_I`, `DejaVAE_P`, `DejaVAE_TF` | Déjà VAE-I, -P, -TF |
| `DejaVAE_P_noCF`, `DejaVAE_I_noCF` | Déjà VAE-P / -I trained without counterfactual reals |

## Data

Download the datasets from their authors:
- GenImage (Zhu et al., 2023)
- Chameleon (Yan et al., 2025)
- Synthbuster (Bammey, 2024; Zenodo 10.5281/zenodo.10066460; `code/pipelines/dl_synthbuster.sh` checks the MD5)
- RAISE-1k (Dang-Nguyen et al., 2015; `code/raise_download.py` takes the CSV the RAISE site provides after registration)

The scripts expect the data under `~/anchor_data/` and write intermediate results to `runs/` next to the code. The
dataset paths are set at the top of `prepare_genimage.py`, `cham_match.py` and `synthbuster_prep.py`. Model weights are
loaded by name from public Hugging Face repositories (Appendix A).

Rebuilding the benchmark variants:
- **Synthbuster with downscaled reals:** `synthbuster_prep.py make <RAISE_DIR>`. The split by RAISE id is listed in
  `data/matching/synthbuster_raise_split.csv`, and the excluded photograph in `synthbuster_excluded.txt`.
- **Synthbuster with cropped reals:** `synthbuster_crop.py make <RAISE_TIFF_DIR>` (after the step above; same ids and split).
- **Size-matched Chameleon and its control:** `cham_match.py make` and `cham_match.py make_ctrl` (seed 17). The resulting
  image sizes are listed in `data/matching/` for checking.

## Paper to code

| Paper | Code | Pipeline |
|---|---|---|
| Sec. 3: training rows, Déjà VAE heads, evaluation | `dejavae.py`, `anchor_pilot.py` | `run_dv2.sh` |
| Sec. 3: baselines (SimLBR, probes) | `anchor_pilot.py` | `run_p2.sh`, `run_p3.sh`, `run_p4.sh`, `run_chameleon.sh` |
| Sec. 3: RIGID, AEROBLADE | `rigid.py`, `aeroblade.py` | `run_baselines.sh`, `run_e3.sh`, `run_rigid_norm.sh` |
| Sec. 3: layer pilot | `dv_layers.py` | |
| Sec. 4: layer curves, copy conditions (Fig. 2) | `dv_layers_full.py` | `run_e1e2e4.sh` |
| Sec. 4: CLIP layer curve | `dv_layers_clip.py` | `run_after_e5b.sh` |
| Sec. 6: processing (Fig. 3), processing-aware training | `dv_resize.py`, `dejavae.py` (`DV_AUG=post`) | `run_dv3.sh` |
| Sec. 6: decoder bank (Fig. 4), BigGAN resolution control | `dv_bank_full.py`, `dv_bank_res.py` | `run_e1e2e4.sh` |
| Sec. 6: GenImage bank with JPEG q95 re-save (Fig. 4), original files (App. G) | `dv_bank_q95.py`, `dv_bank_full.py`, `review_cpu2.py` | `run_review.sh`, `run_review_cpu.sh` |
| Sec. 6: blurred generated images (evasion) | `dv_evasion.py` | `run_review.sh` |
| Sec. 6: copies from SD1's decoder (Table 7) | `dejavae.py` (`DV_AE=...`), `e8_compare.py` | `run_after_e5b.sh` |
| Sec. 7: Chameleon audit and decomposition (Figs. 5, 6) | `cham_match.py`, `e6_decomp.py` | `run_e6.sh`, `run_e6c.sh` |
| Sec. 8: Synthbuster (Fig. 7, Table 1) | `synthbuster_prep.py`, `sb_extra.py` | `run_e5.sh` |
| Sec. 8: cropped reals (Table 2, App. F), photograph-level intervals, downscale factor | `synthbuster_crop.py`, `sb_cluster.py`, `review_downscale.py` | `run_review.sh`, `run_review_cpu.sh` |
| Sec. 8: AEROBLADE without the re-save | `aeroblade.py --set sb_raw` | `run_review.sh` |
| Sec. 8: decoder bank (Table 3) and post-hoc SDXL analysis | `dv_bank_sb.py`, `dv_bank_sdxl.py`, `dv_bank_sb_crop.py` | `run_e5b.sh`, `run_e5b_resume.sh`, `run_review2.sh` |
| Sec. 9: fusion (Fig. 8) | `figures/fig_fusion.py` | |
| App. E: layer and counterfactual ablations (Table 5), pilot exclusion, final layer on Synthbuster (both matchings) | `review_cpu.py`, `review_cpu2.py`, `review_layer_sb.py`, `sb_crop_prefill.py`, `dejavae.py` (`dv-single-nocf`) | `run_review.sh`, `run_review2.sh`, `run_review3.sh`, `run_review_cpu.sh` |
| App. E: linear vs MLP heads; final-layer heads under processing and blur | `review_linear.py`, `dv_resize.py`, `dv_evasion.py` (with `DV_WORK=runs/dv DV_FEAT=cls24`) | `run_review3.sh`, `run_review_cpu.sh` |
| Sec. 8: exact decoder vs best other autoencoder (AUROC margins) | `review_bank_margin.py` | `run_review_cpu.sh` |
| App. I: training-data scale | `scale_report.py` | `run_scale.sh` |
| App. A, Limitations: test-time cost per image | `latency.py` (`latency.py aero 4` for AEROBLADE) | `run_review2.sh` |

Internal names in the code: `dv-single` = Déjà VAE-I, `dv` = Déjà VAE-P, `dv-tf` = Déjà VAE-TF, `lbr_exact` = SimLBR
retrained, `simlbr_official` = SimLBR released, `sup` = supervised probe, `recon` = reconstruction probe. The files
`anchor_pilot.py` and `dv_bank.py` also contain code from an earlier pilot that the paper does not use.

## Licence

Code: MIT (see `LICENSE`). The Figure 1 crops come from Synthbuster (Bammey, 2024; CC BY-NC-SA 4.0) and RAISE-1k
(Dang-Nguyen et al., 2015) and remain under those terms. The scores and lists are derived from GenImage, Chameleon, Synthbuster (CC BY-NC-SA 4.0) and RAISE-1k;
use them under those datasets' terms.
