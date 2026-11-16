#!/bin/zsh
# E5b: decoder-matching bank on Synthbuster; starts when E5 has finished (log runs/e5b.log).
cd "$(dirname "$0")/.."
export DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> runs/e5b.log 2>&1
echo "== $(date) E5b waiting for E5"
until grep -q "E5 FINISHED" runs/e5.log; do sleep 60; done
echo "== $(date) E5b START"
anchor/bin/python dv_bank_sb.py run
echo "== $(date) E5b FINISHED"
