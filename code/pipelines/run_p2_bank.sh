#!/bin/bash
# Reconstruction bank (FLUX 16-ch, VQ, DC-AE) for the format-controlled run runs/p2. Resumable.
set -euo pipefail
cd "$(dirname "$0")"
P=anchor/bin/python; W=runs/p2
exec >> $W/bank.log 2>&1
echo "== $(date) bank start"
$P anchor_pilot.py --work $W recon --name flux --vae ostris/Flex.1-alpha:vae
$P anchor_pilot.py --work $W recon --name vq   --vae kandinsky-community/kandinsky-2-2-decoder:movq
$P anchor_pilot.py --work $W recon --name dcae --vae mit-han-lab/dc-ae-f32c32-in-1.0-diffusers
echo "== $(date) BANK FINISHED"
