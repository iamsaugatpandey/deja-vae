#!/bin/zsh
# E6 control: processing-only fakes (same resize factor + JPEG q75 as the matched set, own aspect, no crop).
set -e
cd "$(dirname "$0")/.."
L=runs/e6c.log; P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> $L 2>&1
echo "== $(date) E6C START"
$P cham_match.py make_ctrl
$P cham_match.py seed_ctrl
echo "== $(date) features (fakes only)"
$P anchor_pilot.py --work runs/cham_c features --batch 16 --preproc simlbr --skip_train
$P anchor_pilot.py --work runs/cham_c score --official genimage
$P anchor_pilot.py --work runs/cham_c score --heads_from runs/p4s
echo "== $(date) Déjà VAE pairs (fakes only)"
$P dejavae.py pairs --set cham_c
echo "== $(date) RIGID dinov3 (fakes only)"
$P rigid.py --work runs/cham_c --backbone dinov3 --seed_from runs/cham_off
echo "== $(date) reports"
$P dejavae.py report --set cham_c
echo "== $(date) E6C FINISHED"
