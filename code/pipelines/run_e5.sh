#!/bin/zsh
# E5: Synthbuster (9 generators x 1000) + RAISE-1k reals geometry-matched per image (synthbuster_prep.py).
# Every evaluation image gets the JPEG q95 re-save, as for GenImage. Waits for raise_download.py to finish.
set -e
cd "$(dirname "$0")/.."
L=runs/e5.log; P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> $L 2>&1
echo "== $(date) E5 START (waiting for RAISE download)"
until grep -q "^done:" runs/raise_dl.log; do sleep 60; done
grep "^done:" runs/raise_dl.log
$P -I synthbuster_prep.py make ~/anchor_data/raise/tiff
echo "== $(date) features"
$P anchor_pilot.py --work runs/sb features --batch 16 --preproc simlbr --skip_train --test_jpeg 95
$P anchor_pilot.py --work runs/sb score --official genimage
$P anchor_pilot.py --work runs/sb score --heads_from runs/p4s
echo "== $(date) Déjà VAE pairs"
$P dejavae.py pairs --set sb
echo "== $(date) RIGID"
$P rigid.py --work runs/sb --test_jpeg 95 --backbone dinov2
$P rigid.py --work runs/sb --test_jpeg 95 --backbone dinov2 --noise_space norm
$P rigid.py --work runs/sb --test_jpeg 95 --backbone dinov3
echo "== $(date) reports"
$P anchor_pilot.py --work runs/sb eval > runs/sb/eval.log 2>&1 || true
$P dejavae.py report --set sb
echo "== $(date) AEROBLADE (subset)"
$P aeroblade.py run --set sb
$P aeroblade.py report --set sb
echo "== $(date) E5 FINISHED"
