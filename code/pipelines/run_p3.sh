#!/bin/bash
# ANCHOR vs SimLBR: rigour run on the format-controlled data (runs/p2). Safe to re-run (skips finished heads).
#  A (CPU, now)       SimLBR exact recipe, linear probes, ANCHOR+SimLBR, beta sensitivity      (single SD VAE)
#  B (CPU, after bank) zero-fake beta selection on held-out autoencoder families (Z2), per-family
#                     singles, all-4 bank, leave-one-family-out banks
#  C (GPU, after bank) features with SimLBR's own preprocessing -> the RELEASED SimLBR head on the same
#                     test images (GenImage JPEG-aligned, GenImage raw, Chameleon)
#  D                  score everything on Chameleon, eval both benchmarks
# Log: runs/p2/p3.log     Progress: python3 check_progress.py --work runs/p2
set -euo pipefail
cd "$(dirname "$0")"
P=anchor/bin/python
W=runs/p2; C=runs/cham_p2
SEEDS="0 1 2"
exec >> $W/p3.log 2>&1
have() { [ -f "$W/scores_$1_seed2.npy" ]; }
train() { local tag=$1; shift
  if have "$tag"; then echo "skip $tag"; else $P anchor_pilot.py --work $W train --seeds $SEEDS "$@" | grep -E "saved"; fi; }
ALL=sd,flux,vq,dcae; A4=sd-flux-vq-dcae
MIX="recon mix_unpaired_matched mix_aligned"

# planned heads (for check_progress)
{
  printf "%s\n" sup lbr recon mix_unpaired mix_unpaired_matched mix_aligned recon+fakes mix_unpaired_matched+fakes mix_aligned+fakes
  printf "%s\n" lbr_exact sup+lin lbr_exact+lin
  for m in $MIX; do echo "$m+lbr"; done
  for b in b0.25-0.75 b0.5-1 b1-2 b1-3; do echo "mix_aligned~$b"; echo "mix_unpaired_matched~$b"; done
  for f in flux vq dcae; do for m in $MIX; do echo "$m@$f"; done; done
  for m in $MIX; do echo "$m@$A4"; echo "$m+lbr@$A4"; echo "$m+fakes@$A4"; done
  for out in sd flux vq dcae; do keep=$(echo sd flux vq dcae | tr ' ' '\n' | grep -v "^$out$" | paste -sd- -); for m in $MIX; do echo "$m@$keep"; done; done
  for m in mix_aligned mix_unpaired_matched; do echo "$m~z2"; echo "$m~z2@$A4"; done
} > $W/planned_heads.txt

echo "== $(date) A: single-VAE heads"
train lbr_exact --method lbr_exact --dropout 0.3
train sup+lin --method sup --linear
train lbr_exact+lin --method lbr_exact --linear
for m in $MIX; do train $m+lbr --method $m --with_lbr; done
for r in 0.25-0.75 0.5-1 1-2 1-3; do lo=${r%-*}; hi=${r#*-}
  for m in mix_aligned mix_unpaired_matched; do train "$m~b$r" --method $m --tag "b$r" --b_min $lo --b_max $hi; done
done

echo "== $(date) waiting for the reconstruction bank (run_p2_bank.sh)"
until grep -q "BANK FINISHED" $W/bank.log 2>/dev/null; do
  pgrep -f "run_p2_bank\.sh" > /dev/null || { echo "bank job not running and not finished: start ./run_p2_bank.sh"; exit 1; }
  sleep 60
done

# C runs on the GPU in the background while B trains on the CPU
official_features() {   # DIR MANIFEST_SOURCE [features args...]
  local d=$1 src=$2; shift 2; mkdir -p $d; [ -e $d/manifest.csv ] || cp $src $d/manifest.csv
  $P anchor_pilot.py --work $d features --batch 16 --preproc simlbr --skip_train "$@"
}
(
  echo "== $(date) C: SimLBR-preprocessed features (GenImage JPEG-aligned, Chameleon, GenImage raw)"
  official_features runs/p2_off $W/manifest.csv --jpeg_prob 1.0 --test_jpeg 95
  official_features runs/cham_off runs/cham/manifest.csv
  official_features runs/p2raw_off $W/manifest.csv --jpeg_prob 1.0
  echo "== $(date) C done"
) > $W/p3_official.log 2>&1 &
CPID=$!

echo "== $(date) B: zero-fake beta selection (held-out autoencoder families)"
[ -f $W/select_beta.json ] || $P anchor_pilot.py --work $W select --bank $ALL
read EL EH DL DH < <($P -c "import json;c=json.load(open('$W/select_beta.json'))['chosen'];print(*c['mix_aligned'],*c['mix_unpaired_matched'])")
echo "chosen beta: E U($EL,$EH)  D* U($DL,$DH)"
train mix_aligned~z2 --method mix_aligned --tag z2 --b_min $EL --b_max $EH
train mix_unpaired_matched~z2 --method mix_unpaired_matched --tag z2 --b_min $DL --b_max $DH
train mix_aligned~z2@$A4 --method mix_aligned --tag z2 --b_min $EL --b_max $EH --bank $ALL
train mix_unpaired_matched~z2@$A4 --method mix_unpaired_matched --tag z2 --b_min $DL --b_max $DH --bank $ALL
echo "== $(date) B: per-family, all-4 and leave-one-family-out banks"
for f in flux vq dcae; do for m in $MIX; do train $m@$f --method $m --bank $f; done; done
for m in $MIX; do
  train $m@$A4 --method $m --bank $ALL
  train $m+lbr@$A4 --method $m --bank $ALL --with_lbr
  train $m+fakes@$A4 --method $m --bank $ALL --with_fakes
done
for out in sd flux vq dcae; do
  keep=$(echo sd flux vq dcae | tr ' ' '\n' | grep -v "^$out$" | paste -sd, -)
  for m in $MIX; do train "$m@${keep//,/-}" --method $m --bank $keep; done
done

echo "== $(date) waiting for C (GPU features)"
wait $CPID
echo "== $(date) D: released SimLBR head + scoring + eval"
$P anchor_pilot.py --work runs/p2_off score --official genimage --out $W
$P anchor_pilot.py --work runs/cham_off score --official genimage --out $C
$P anchor_pilot.py --work runs/p2raw_off score --official genimage
$P anchor_pilot.py --work $C score --heads_from $W
[ -f $W/eval_v1.log ] || cp $W/eval.log $W/eval_v1.log
[ -f $C/eval_v1.log ] || cp $C/eval.log $C/eval_v1.log
$P anchor_pilot.py --work $W eval > $W/eval.log 2>&1
$P anchor_pilot.py --work $C eval > $C/eval.log 2>&1
echo "== $(date) P3 FINISHED (results: $W/eval.log, $C/eval.log)"
