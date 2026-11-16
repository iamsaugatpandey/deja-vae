#!/bin/zsh
# Queued after run_e3.sh: RIGID with noise on the normalised input (the stronger convention in the subset
# sanity check) on ALL eval rows, so the main tables use the strongest version of the baseline.
cd "$(dirname "$0")/.."
L=runs/baselines.log; PY=anchor/bin/python
while pgrep -f run_e3.sh > /dev/null; do sleep 60; done
echo "== $(date) RIGID-NORM START" >> $L
$PY rigid.py --work runs/p4s --test_jpeg 95 --backbone dinov2 --noise_space norm >> $L 2>&1
$PY rigid.py --work runs/cham_off --backbone dinov2 --noise_space norm >> $L 2>&1
echo "== $(date) RIGID-NORM FINISHED" >> $L
