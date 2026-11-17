#!/bin/zsh
# After run_review.sh: crop-matched decoder bank, final-layer Deja VAE (v1) on Synthbuster, latency. Log: runs/review.log
cd "$(dirname "$0")/.."
P=anchor/bin/python; export PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> runs/review.log 2>&1
until grep -q "REVIEW FINISHED" runs/review.log; do sleep 60; done
echo "== $(date) 6. Synthbuster decoder bank with crop-matched reals"
DV_FEAT=mid $P dv_bank_sb_crop.py run || echo "!! bank_crop failed"
echo "== $(date) 7. final-layer Deja VAE (v1) on Synthbuster"
DV_WORK=runs/dv DV_FEAT=cls24 $P dejavae.py pairs --set sb && DV_WORK=runs/dv DV_FEAT=cls24 $P dejavae.py report --set sb || echo "!! v1 sb failed"
echo "== $(date) 8. latency"
DV_FEAT=mid $P latency.py || echo "!! latency failed"
DV_FEAT=mid $P latency.py aero 4 || echo "!! latency (AEROBLADE, batch 4) failed"   # batch 16 swaps on 16 GB
echo "== $(date) REVIEW2 FINISHED"
