#!/bin/bash
# Déjà VAE pilot v2: same as run_dv.sh but DINOv3 CLS from middle layers 8-18 (layer diagnostic,
# runs/dv/layers/report.txt). Writes to runs/dv2 (v1 in runs/dv stays untouched). Log: runs/dv2/dv.log.
# Safe to re-run: every step resumes.   Status: ./dv_status.sh 2
set -euo pipefail
cd "$(dirname "$0")/.."
P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid
mkdir -p runs/dv2
exec >> runs/dv2/dv.log 2>&1
echo "== $(date) 1. training pairs";  $P dejavae.py pairs --set train
echo "== $(date) 2. heads";           $P dejavae.py train
echo "== $(date) 3. GenImage pairs";  $P dejavae.py pairs --set genimage
$P dejavae.py report --set genimage
echo "== $(date) 4. Chameleon pairs"; $P dejavae.py pairs --set cham
$P dejavae.py report --set cham
echo "== $(date) DV FINISHED (runs/dv2/report_genimage.txt, runs/dv2/report_cham.txt)"
