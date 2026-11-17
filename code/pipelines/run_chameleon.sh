#!/bin/bash
# Chameleon (AIDE) as an extra, test-only benchmark. Waits for run_pilot.sh to finish
# (so the two never compete for the 16 GB of memory), then:
#   DINOv3 ViT-L features for Chameleon -> score every head trained in runs/p1 -> eval.
# Calibration: 30% of Chameleon reals (runs/cham/manifest.csv, split=calib) set each threshold.
# Log: runs/cham/pipeline.log      Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
P=anchor/bin/python
W=runs/cham
mkdir -p $W
exec >> $W/pipeline.log 2>&1
echo "== $(date) waiting for the GenImage pipeline (runs/p1) to finish"
until grep -q "PIPELINE FINISHED" runs/p1/pipeline.log 2>/dev/null; do
  pgrep -f "run_pilot\.sh" > /dev/null || { echo "run_pilot.sh is not running and has not finished: restart it first"; exit 1; }
  sleep 60
done
echo "== $(date) features (Chameleon, DINOv3 ViT-L/16)"
$P anchor_pilot.py --work $W features --batch 16
echo "== $(date) score heads from runs/p1"
$P anchor_pilot.py --work $W score --heads_from runs/p1
echo "== $(date) eval"
$P anchor_pilot.py --work $W eval > $W/eval.log 2>&1
echo "== $(date) CHAMELEON FINISHED (results: $W/eval.log)"
