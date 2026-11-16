#!/bin/zsh
# E6: size-matched Chameleon. Build matched fakes, audit metadata, seed real rows, recompute fakes for
# every detector (SimLBR released/retrained, sup, recon, Déjà VAE v2, RIGID x3, AEROBLADE subset), report.
set -e
cd "$(dirname "$0")/.."
L=runs/e6.log; P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> $L 2>&1
echo "== $(date) E6 START"
$P cham_match.py make
$P cham_match.py audit
$P cham_match.py seed
echo "== $(date) features (fakes only)"
$P anchor_pilot.py --work runs/cham_m features --batch 16 --preproc simlbr --skip_train
$P anchor_pilot.py --work runs/cham_m score --official genimage
$P anchor_pilot.py --work runs/cham_m score --heads_from runs/p4s
echo "== $(date) Déjà VAE pairs (fakes only)"
$P dejavae.py pairs --set cham_m
echo "== $(date) RIGID (fakes only)"
$P rigid.py --work runs/cham_m --backbone dinov2 --seed_from runs/cham_off
$P rigid.py --work runs/cham_m --backbone dinov2 --noise_space norm --seed_from runs/cham_off
$P rigid.py --work runs/cham_m --backbone dinov3 --seed_from runs/cham_off
echo "== $(date) AEROBLADE (subset)"
$P aeroblade.py run --set cham_m
echo "== $(date) reports"
$P anchor_pilot.py --work runs/cham_m eval > runs/cham_m/eval.log 2>&1 || true
$P dejavae.py report --set cham_m
$P aeroblade.py report --set cham_m
echo "== $(date) E6 FINISHED"
