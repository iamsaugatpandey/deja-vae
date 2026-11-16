#!/bin/zsh
# E3 sanity + AEROBLADE. RIGID on the fixed GenImage subset WITHOUT the JPEG re-save, both noise conventions
# (the paper does not say which), then AEROBLADE on both benchmarks, then the subset reports.
cd "$(dirname "$0")/.."
L=runs/e3.log; PY=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
echo "== $(date) E3 START" >> $L
for ns in pixel norm; do
  $PY rigid.py --work runs/p4s --backbone dinov2 --idx runs/p4s/aeroblade/idx.npy --noise_space $ns --suffix "~raw" >> $L 2>&1
  $PY rigid.py --work runs/p4s --backbone dinov2 --idx runs/p4s/aeroblade/idx.npy --noise_space $ns --suffix "~q95" --test_jpeg 95 >> $L 2>&1
done
for s in genimage cham; do
  $PY aeroblade.py run --set $s >> $L 2>&1
  $PY aeroblade.py report --set $s >> $L 2>&1
done
echo "== $(date) E3 FINISHED" >> $L
