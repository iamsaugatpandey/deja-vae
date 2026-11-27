#!/bin/zsh
# E5: false alarms on VISION phone photos. Waits for the download and the EvalGEN jobs (GPU), then scores everything.
cd "$(dirname "$0")/.."
P=anchor/bin/python
export DV_WORK=runs/dv2 DV_FEAT=mid PYTORCH_ENABLE_MPS_FALLBACK=1
while pgrep -f "e5_download.sh|run_eg.sh|run_gi.sh" >/dev/null; do sleep 60; done
echo "== $(date) manifest"; $P -I e5_prep.py || { echo "e5 prep failed"; exit 1; }
echo "== $(date) ours on e5"
$P anchor_pilot.py --work runs/e5 features --batch 16 --preproc simlbr --skip_train --test_jpeg 95 && \
$P anchor_pilot.py --work runs/e5 score --official genimage && \
$P dejavae.py pairs --set e5 || echo "ours e5 failed"
for d in bfree dda npr alignedforensics spai; do
  echo "== $(date) E1 $d on VISION"
  $P -I e1_score.py $d --sets e5 > runs/e1/${d}_e5.log 2>&1 || echo "$d e5 failed"
  grep -E "done|FINISHED" runs/e1/${d}_e5.log
done
echo "== $(date) report"; $P e5_report.py > /dev/null 2>&1 || echo "e5 report failed"
echo "== $(date) E5 FINISHED"
