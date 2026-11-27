#!/bin/zsh
# EvalGEN (2024-25 generators) with matched RAISE reals: the paper's methods, then the five released detectors, then report + stats.
cd "$(dirname "$0")/.."
P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "run_e4b.sh|evalgen_prep.py" >/dev/null; do sleep 30; done
[ -f runs/eg_crop/manifest.csv ] || { echo "no EvalGEN manifest; stopping"; exit 1; }
for s in eg eg_crop; do
  echo "== $(date) ours on $s"
  $P anchor_pilot.py --work runs/$s features --batch 16 --preproc simlbr --skip_train --test_jpeg 95 && \
  $P anchor_pilot.py --work runs/$s score --official genimage && \
  $P dejavae.py pairs --set $s && \
  $P dejavae.py report --set $s > /dev/null || echo "ours $s failed"
done
for d in bfree dda npr alignedforensics spai; do
  echo "== $(date) E1 $d on EvalGEN"
  $P -I e1_score.py $d --sets eg_q95 eg_crop_q95 > runs/e1/${d}_eg.log 2>&1 || echo "E1 $d EvalGEN failed"
  grep -E "done|FINISHED" runs/e1/${d}_eg.log
done
echo "== $(date) report + stats"
$P e1_report.py > /dev/null 2>&1 || echo "report failed"
$P e1_stats.py > runs/e1/stats.log 2>&1 || echo "stats failed"
echo "== $(date) EG FINISHED"
