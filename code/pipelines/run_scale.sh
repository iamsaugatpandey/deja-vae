#!/bin/bash
# Data-scale curve for SimLBR / supervised baselines: N reals + N fakes, N = 1250, 2500, 5000 (10k = runs/p4s).
# Reuses runs/p4s cached features (symlinks), so no image is re-processed. Heads train on CPU (light).
# Log: runs/scale/scale.log. Report: runs/scale/report.txt. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
P=anchor/bin/python
mkdir -p runs/scale
exec >> runs/scale/scale.log 2>&1
echo "== $(date) build subsets"
$P - <<'PY'
import csv, json, os, numpy as np
from pathlib import Path
src = Path("runs/p4s"); rows = list(csv.DictReader(open(src / "manifest.csv")))
rng = np.random.default_rng(123)
perm = {lab: rng.permutation([i for i, r in enumerate(rows) if r["split"] == "train" and r["label"] == lab]) for lab in ("0", "1")}
for n in (1250, 2500, 5000):
    W = Path(f"runs/scale/n{n}"); W.mkdir(parents=True, exist_ok=True)
    keep = set(perm["0"][:n].tolist()) | set(perm["1"][:n].tolist())   # nested: n1250 within n2500 within n5000
    out = [dict(r, split="pool") if r["split"] == "train" and i not in keep else r for i, r in enumerate(rows)]
    with open(W / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
    for fn in ("cls.npy", "cls_recon.npy", "cls_degraded.npy"):
        if not (W / fn).exists(): os.symlink(os.path.abspath(src / fn), W / fn)
    json.dump(json.load(open(src / "features_meta.json")), open(W / "features_meta.json", "w"), indent=1)
    json.dump(list(range(len(rows))), open(W / "features_done.json", "w"))
    n_tr = sum(r["split"] == "train" for r in out); print(f"n{n}: {n_tr} training rows")
PY
for n in 1250 2500 5000; do
  W=runs/scale/n$n
  [ -f $W/scores_lbr_exact~n${n}_seed2.npy ] || nice -n 10 $P anchor_pilot.py --work $W train --method lbr_exact --dropout 0.3 --tag n$n | grep saved
  [ -f $W/scores_sup~n${n}_seed2.npy ] || nice -n 10 $P anchor_pilot.py --work $W train --method sup --tag n$n | grep saved
  nice -n 10 $P anchor_pilot.py --work runs/cham_off score --heads_from $W
done
$P scale_report.py
echo "== $(date) SCALE FINISHED (runs/scale/report.txt)"
