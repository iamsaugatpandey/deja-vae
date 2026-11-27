#!/bin/zsh
cd "$(dirname "$0")/.."
echo "== $(date) zhou sd1 (re-run: output-name bug)"; DV_FEAT=mid anchor/bin/python review_copies.py --ae sd1 || echo "zhou sd1 failed"
echo "== $(date) zhou compare"; anchor/bin/python zhou_compare.py || echo "zhou compare failed"
echo "== $(date) E4B FINISHED"
