#!/bin/zsh
# E5b resumed 16 Nov: the consistency decoder (batch 4) exhausted 16 GB RAM and thrashed swap; rerun with batch 1.
# The six finished autoencoders are kept (dv_bank_sb.py skips existing phi_R_*.npz).
cd "$(dirname "$0")/.."
export DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
exec >> runs/e5b.log 2>&1
echo "== $(date) E5b START (resumed: consistency decoder, batch 1, fp16)"
anchor/bin/python dv_bank_sb.py run && echo "== $(date) E5b FINISHED" || echo "== $(date) E5b FAILED (see Traceback above)"
