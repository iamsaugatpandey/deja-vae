#!/bin/zsh
# CPU-only analyses on the outputs of run_review.sh / run_review2.sh (no GPU, a few minutes each).
cd "$(dirname "$0")/.."
P=anchor/bin/python; export DV_WORK=runs/dv2 DV_FEAT=mid
$P review_cpu.py                     # layer ablation (final layer vs layers 8-18), pilot exclusion
$P review_cpu2.py                    # q95 bank attribution; Deja VAE-I counterfactual ablation
$P review_downscale.py               # Synthbuster: score vs downscale factor of the matched reals
$P review_layer_sb.py                # Synthbuster: final layer vs layers 8-18, both matchings (needs run_review3.sh)
$P review_linear.py                  # linear vs MLP heads at the final layer and at layers 8-18
DV_FEAT=mid $P review_bank_margin.py # exact decoder vs best other autoencoder, both matchings
$P sb_extra.py sb; $P sb_extra.py sb_crop      # extra paired comparisons, both matchings
$P sb_cluster.py; $P sb_cluster.py pergen      # Synthbuster intervals resampled by RAISE photograph
$P dejavae.py report --set sb_crop; $P aeroblade.py report --set sb_raw
