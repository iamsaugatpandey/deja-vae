#!/bin/zsh
# E4 GPU jobs, one after the other: Zhou test with SD1-decoder copies (+ ft-MSE re-run), then the Synthbuster bank on layers 8-18.
cd "$(dirname "$0")/.."
P=anchor/bin/python
echo "== $(date) zhou sd1";   DV_FEAT=mid $P review_copies.py --ae sd1   || echo "zhou sd1 failed"
echo "== $(date) zhou ftmse"; DV_FEAT=mid $P review_copies.py --ae ftmse || echo "zhou ftmse failed"
echo "== $(date) zhou compare"; $P zhou_compare.py || echo "zhou compare failed"
export BANK_LAYERS=8,10,12,14,16,18 SB_BANK_OUT=runs/sb/bank_l8 SB_BANK_CROP_OUT=runs/sb/bank_crop_l8 DV_FEAT=mid
echo "== $(date) bank 8-18 downscaled"; $P dv_bank_sb.py run      || echo "bank l8 failed"
echo "== $(date) bank 8-18 cropped";    $P dv_bank_sb_crop.py run || echo "bank crop l8 failed"
echo "== $(date) E4 FINISHED"
