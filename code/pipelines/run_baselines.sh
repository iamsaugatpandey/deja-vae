#!/bin/bash
# Training-free baselines on every calib/test row of GenImage (p2, JPEG q95) and Chameleon.
#   RIGID (TMLR): DINOv2 ViT-L/14 as published, and DINOv3 ViT-L/16 for backbone parity.
# Log: runs/baselines.log. Safe to re-run (finished scores are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
P=anchor/bin/python
exec >> runs/baselines.log 2>&1
for bb in dinov2 dinov3; do
  echo "== $(date) RIGID $bb GenImage"; $P rigid.py --work runs/p4s --test_jpeg 95 --backbone $bb
  echo "== $(date) RIGID $bb Chameleon"; $P rigid.py --work runs/cham_off --backbone $bb
done
echo "== $(date) BASELINES FINISHED"
