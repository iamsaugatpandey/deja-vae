#!/bin/zsh
# Final-layer heads (runs/dv, DV_FEAT=cls24) under the processing grid, the blur evasion test, and crop-matched
# Synthbuster. Log: runs/review.log
cd "$(dirname "$0")/.."
P=anchor/bin/python; export DV_WORK=runs/dv DV_FEAT=cls24
exec >> runs/review.log 2>&1
echo "== $(date) 9. final-layer heads: processing grid"
$P dv_resize.py post && $P dv_resize.py report && $P dv_resize.py sizes || echo "!! v1 resize failed"
echo "== $(date) 10. final-layer heads: blur evasion"
$P dv_evasion.py || echo "!! v1 evasion failed"
echo "== $(date) 11. final-layer heads: crop-matched Synthbuster"
$P sb_crop_prefill.py && $P dejavae.py pairs --set sb_crop && $P dejavae.py report --set sb_crop || echo "!! v1 sb_crop failed"
echo "== $(date) REVIEW3 FINISHED"
