#!/bin/zsh
# Robustness controls (16 Nov). Log: runs/review.log
cd "$(dirname "$0")/.."
L=runs/review.log; P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> $L 2>&1
echo "== $(date) REVIEW START (waiting for latency.py)"
while pgrep -f latency.py >/dev/null; do sleep 20; done
echo "== $(date) 1. GenImage decoder bank with JPEG q95 on both classes"
$P dv_bank_q95.py || echo "!! bank_q95 failed"
echo "== $(date) 2. Deja VAE-I without counterfactual rows (train + reports)"
$P dejavae.py train && for s in genimage cham sb; do $P dejavae.py report --set $s > /dev/null; done || echo "!! nocf failed"
echo "== $(date) 3. evasion (blurred fakes)"
$P dv_evasion.py || echo "!! evasion failed"
echo "== $(date) 4. Synthbuster with crop-matched reals"
$P -I synthbuster_crop.py make ~/anchor_data/raise/tiff && \
$P anchor_pilot.py --work runs/sb_crop features --batch 16 --preproc simlbr --skip_train --test_jpeg 95 && \
$P anchor_pilot.py --work runs/sb_crop score --official genimage && \
$P anchor_pilot.py --work runs/sb_crop score --heads_from runs/p4s && \
$P dejavae.py pairs --set sb_crop && \
$P dejavae.py report --set sb_crop || echo "!! sb_crop failed"
echo "== $(date) 5. AEROBLADE on Synthbuster without the re-save"
$P aeroblade.py run --set sb_raw || echo "!! aeroblade sb_raw failed"
echo "== $(date) REVIEW FINISHED"
