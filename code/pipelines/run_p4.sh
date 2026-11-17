#!/bin/bash
# Preprocessing control: why does the RELEASED SimLBR head beat our retrained SimLBR on Chameleon
# (84.0 vs 77.0 TPR@5%) while they tie on GenImage? Retrain every key head on features made with
# SimLBR's own preprocessing (256x256 squash resize + SimLBR normalisation), same 10k/10k SD1.4 data,
# same JPEG alignment as runs/p2. If our lbr_exact reaches ~84 on Chameleon, the gap was preprocessing;
# if not, it is training-data scale (released head: full 162k/162k SD1.4 train).
#   runs/p4s     train + GenImage (JPEG-aligned) eval   (calib/test features reused from runs/p2_off)
#   runs/cham_off  Chameleon, same preprocessing
#   runs/p2raw_off raw GenImage (paper protocol)
# Log: runs/p4s/p4.log.  Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
P=anchor/bin/python
W=runs/p4s; SEEDS="0 1 2"
mkdir -p $W
exec >> $W/p4.log 2>&1

if [ ! -f $W/features_meta.json ]; then
  echo "== $(date) seeding $W from runs/p2_off (calib/test rows)"
  cp runs/p2/manifest.csv $W/
  cp runs/p2_off/cls.npy runs/p2_off/cls_degraded.npy runs/p2_off/cls_recon.npy $W/
  $P - <<'PY'
import json, csv
W = "runs/p4s"
m = json.load(open("runs/p2_off/features_meta.json")); m.pop("skip_train")
json.dump(m, open(f"{W}/features_meta.json", "w"), indent=1)
rows = list(csv.DictReader(open(f"{W}/manifest.csv")))
json.dump([i for i, r in enumerate(rows) if r["split"] != "train"], open(f"{W}/features_done.json", "w"))
PY
fi
echo "== $(date) features for the 20k training rows (SimLBR preprocessing, SD VAE reconstructions)"
$P anchor_pilot.py --work $W features --batch 16 --preproc simlbr --jpeg_prob 1.0 --test_jpeg 95

have() { [ -f "$W/scores_$1_seed2.npy" ]; }
train() { local tag=$1; shift
  if have "$tag"; then echo "skip $tag"; else $P anchor_pilot.py --work $W train --seeds $SEEDS "$@" | grep -E "saved"; fi; }
MIX="recon mix_unpaired_matched mix_aligned"
{ printf "%s\n" sup lbr lbr_exact $MIX mix_aligned~b0.25-0.75
  for m in $MIX; do echo "$m+lbr"; echo "$m+fakes"; done; } > $W/planned_heads.txt
echo "== $(date) train"
train lbr_exact --method lbr_exact --dropout 0.3
train lbr --method lbr
train sup --method sup
for m in $MIX; do train $m --method $m; done
for m in $MIX; do train $m+lbr --method $m --with_lbr; done
for m in $MIX; do train $m+fakes --method $m --with_fakes; done
train mix_aligned~b0.25-0.75 --method mix_aligned --tag b0.25-0.75 --b_min 0.25 --b_max 0.75

echo "== $(date) score + eval"
cp runs/p2/scores_simlbr_official_seed*.npy runs/p2/degscores_simlbr_official_seed*.npy $W/
[ -f runs/cham_off/scores_simlbr_official_seed2.npy ] || $P anchor_pilot.py --work runs/cham_off score --official genimage
for d in runs/cham_off runs/p2raw_off; do $P anchor_pilot.py --work $d score --heads_from $W; done
for d in $W runs/cham_off runs/p2raw_off; do $P anchor_pilot.py --work $d eval > $d/eval.log 2>&1; done
echo "== $(date) P4 FINISHED (results: $W/eval.log, runs/cham_off/eval.log, runs/p2raw_off/eval.log)"
