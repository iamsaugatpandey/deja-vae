#!/bin/zsh
# Queue after E5b: E1c (CLIP layer curve, log runs/e1c.log), then E8 (Déjà VAE retrained with SD1's original decoder,
# work folder runs/dv4, log runs/e8.log). Predictions stated before the run (paper, Sec. 3.4).
cd "$(dirname "$0")/.."
P=anchor/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "== $(date) waiting for E5b" >> runs/e1c.log
until grep -q "E5b FINISHED" runs/e5b.log 2>/dev/null; do sleep 60; done
{ echo "== $(date) E1c START"; DV_FEAT=mid $P dv_layers_clip.py run; echo "== $(date) E1c FINISHED"; } >> runs/e1c.log 2>&1
export DV_WORK=runs/dv4 DV_FEAT=mid DV_AE=stable-diffusion-v1-5/stable-diffusion-v1-5:vae
{
  echo "== $(date) E8 START (AE=$DV_AE)"
  echo "== $(date) 1. training pairs";   $P dejavae.py pairs --set train
  echo "== $(date) 2. heads";            $P dejavae.py train
  echo "== $(date) 3. GenImage pairs";   $P dejavae.py pairs --set genimage
  echo "== $(date) 4. Synthbuster pairs"; $P dejavae.py pairs --set sb
  echo "== $(date) 5. Chameleon pairs";  $P dejavae.py pairs --set cham
  echo "== $(date) 6. reports"
  for s in genimage sb cham; do $P dejavae.py report --set $s > /dev/null; done
  echo "== $(date) E8 FINISHED"
} >> runs/e8.log 2>&1
