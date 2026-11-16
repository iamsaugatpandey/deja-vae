#!/bin/bash
# Format-controlled re-run ("does the result survive removing the JPEG/PNG shortcut?").
#   train: every training image (reals, reconstructions, degraded reals, fakes) JPEG re-saved, q ~ U[70,100]
#   test:  every GenImage calib/test image (real AND fake) re-saved as JPEG q=95; 2k real + 2k fake per generator
#   then the same heads are scored on Chameleon (runs/cham_p2, features shared with runs/cham).
# Log: runs/p2/pipeline.log   Progress: python3 check_progress.py --work runs/p2   Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
P=anchor/bin/python
W=runs/p2
C=runs/cham_p2
G=data/genimage
SEEDS="0 1 2"
mkdir -p $W $C
echo 3 > $W/pipeline_seeds.txt
exec >> $W/pipeline.log 2>&1
printf "%s\n" sup lbr recon mix_unpaired mix_unpaired_matched mix_aligned \
  recon+fakes mix_unpaired_matched+fakes mix_aligned+fakes > $W/planned_heads.txt

have() { [ -f "$W/scores_$1_seed2.npy" ]; }
train() { local tag=$1; shift
  if have "$tag"; then echo "skip $tag (done)"; else $P anchor_pilot.py --work $W train --seeds $SEEDS "$@" | grep -E "step 2000/|saved"; fi; }

if [ ! -f $W/manifest.csv ]; then
  echo "== $(date) manifest"
  ENTRIES=(--entry $G/stable_diffusion_v_1_4/train/nature,real,train,imagenet_sd14,10000
           --entry $G/stable_diffusion_v_1_4/train/ai,fake,train,stable_diffusion_v_1_4,10000)
  for g in ADM BigGAN glide Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 VQDM wukong; do
    ENTRIES+=(--entry $G/$g/test/nature,real,test,$g,2000 --entry $G/$g/test/ai,fake,test,$g,2000)
  done
  $P anchor_pilot.py --work $W manifest --calib_frac 0.3 "${ENTRIES[@]}"
fi
echo "== $(date) features (all training images JPEG; all test images JPEG q95)"
$P anchor_pilot.py --work $W features --batch 16 --jpeg_prob 1.0 --test_jpeg 95
echo "== $(date) train"
for m in sup lbr recon mix_unpaired mix_unpaired_matched mix_aligned; do train $m --method $m; done
for m in recon mix_unpaired_matched mix_aligned; do train $m+fakes --method $m --with_fakes; done
echo "== $(date) eval (GenImage, format-aligned)"
$P anchor_pilot.py --work $W eval > $W/eval.log 2>&1
echo "== $(date) Chameleon"
for f in manifest.csv features_meta.json features_done.json cls.npy cls_recon.npy cls_degraded.npy; do
  [ -e $C/$f ] || ln -s ../cham/$f $C/$f
done
$P anchor_pilot.py --work $C score --heads_from $W
$P anchor_pilot.py --work $C eval > $C/eval.log 2>&1
echo "== $(date) P2 FINISHED (results: $W/eval.log, $C/eval.log)"
