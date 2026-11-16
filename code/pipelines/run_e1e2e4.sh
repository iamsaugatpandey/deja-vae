#!/bin/zsh
# E1+E2 (layer curves, two backbones, four copy conditions) then E4 (autoencoder bank + attribution), full n.
cd "$(dirname "$0")/.."
L=runs/e124.log; P=anchor/bin/python
export DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> $L 2>&1
echo "== $(date) E1E2 START"
$P dv_layers_full.py run
echo "== $(date) E4 START"
$P dv_bank_full.py run
echo "== $(date) E124 FINISHED"
