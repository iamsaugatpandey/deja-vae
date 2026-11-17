#!/bin/bash
# Déjà VAE v3 = v2 (DINOv3 CLS layers 8-18) + in-the-wild processing on every zero-fake training row:
# optional upload JPEG q60-90 (p=0.5) -> resize x{0.5..1.5} (p=0.7) -> native crop, identical for a
# row's real / simulated fake / blur counterfactual. Test features are unchanged, so they are reused
# from runs/dv2 (symlinks); only the training pairs are recomputed. Log: runs/dv3/dv.log.
# Safe to re-run.   Status: ./dv_status.sh 3
set -euo pipefail
cd "$(dirname "$0")/.."
P=anchor/bin/python
export DV_WORK=runs/dv3 DV_FEAT=mid DV_AUG=post
mkdir -p runs/dv3
exec >> runs/dv3/dv.log 2>&1
for f in runs/dv2/genimage_* runs/dv2/cham_*; do [ -e runs/dv3/$(basename $f) ] || ln -s "$PWD/$f" runs/dv3/; done
echo "== $(date) 1. training pairs (processed)";  $P dejavae.py pairs --set train
echo "== $(date) 2. heads";                       $P dejavae.py train
echo "== $(date) 3. reports (GenImage, Chameleon; reused v2 test features)"
$P dejavae.py report --set genimage > /dev/null
$P dejavae.py report --set cham > /dev/null
echo "== $(date) 4. post-processing grid + Chameleon size groups"
$P dv_resize.py sizes > /dev/null
$P dv_resize.py post
echo "== $(date) DV FINISHED (runs/dv3/report_genimage.txt, report_cham.txt, resize/post.txt, resize/sizes.txt)"
