#!/bin/zsh
# Size-matched GenImage + processing-only control: the paper's methods, the five released detectors, then report, stats, figure.
cd "$(dirname "$0")/.."
P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "run_eg.sh" >/dev/null; do sleep 60; done
for s in gi_c gi_m; do
  echo "== $(date) ours on $s"
  $P anchor_pilot.py --work runs/$s features --batch 16 --preproc simlbr --skip_train --test_jpeg 95 && \
  $P anchor_pilot.py --work runs/$s score --official genimage && \
  $P dejavae.py pairs --set $s || echo "ours $s failed"
done
for d in bfree dda npr alignedforensics spai; do
  echo "== $(date) E1 $d on size-matched GenImage"
  $P -I e1_score.py $d --sets gi_c gi_m > runs/e1/${d}_gi.log 2>&1 || echo "E1 $d gi failed"
  grep -E "done|FINISHED" runs/e1/${d}_gi.log
done
echo "== $(date) report + stats + figure"
$P e1_report.py > /dev/null 2>&1 || echo "report failed"
$P e1_stats.py > runs/e1/stats.log 2>&1 || echo "stats failed"
$P paper/figures/fig_e1_slope.py > /dev/null 2>&1 || echo "figure failed"
echo "== $(date) GI FINISHED"
