"""Review: final-layer heads (runs/dv) on crop-matched Synthbuster. sb_crop has the same generated images as sb, at the
same row indices, so their (image, copy) features are copied from runs/dv/sb_* and marked done; `dejavae.py pairs
--set sb_crop` then computes only the cropped reals.   DV_WORK=runs/dv DV_FEAT=cls24 anchor/bin/python sb_crop_prefill.py
"""
import os, sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import dejavae as dv

a, _ = dv.rows_of("sb"); b, _ = dv.rows_of("sb_crop")
fk = [i for i in range(len(a)) if int(a[i]["label"]) == 1]
assert all(a[i]["path"] == b[i]["path"] for i in fk) and len(a) == len(b)
for k in ("img", "deg"):
    for s in "ab":
        src, dst = dv.W / f"sb_{k}_{s}.npy", dv.W / f"sb_crop_{k}_{s}.npy"
        if not dst.exists():
            shutil.copyfile(src, dst)
done_p = dv.W / "sb_crop_done.json"
if not done_p.exists():
    json.dump(fk, open(done_p, "w"))
print(f"prefilled {len(fk)} generated rows of sb_crop from sb in {dv.W}")
