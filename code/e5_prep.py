"""E5: false alarms on everyday real photographs. Manifest of the VISION subset (Shullani et al. 2017, CC BY-SA 4.0).

VISION holds photographs from 35 phones and tablets (11 brands, 2012-2016). Each "nat" photograph also exists as shared
through Facebook (high and low quality) and WhatsApp, as downloaded back from those services by the dataset's authors.
Subset (e5_download.sh, ~/anchor_data/e5/vision_subset.txt, seed 41): per device 20 natural scenes in all four versions
and 10 "flat" photographs (walls, skies: low texture). Every image is real (label 0, split "test").
    anchor/bin/python -I e5_prep.py   -> runs/e5/manifest.csv (+ device, scene columns), runs/e5/idx_all.npy
"""
import csv, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V = Path.home() / "anchor_data" / "e5" / "vision"


def main():
    from PIL import Image
    rows, bad = [], []
    for p in sorted(V.rglob("*.jpg")):
        m = re.search(r"(D\d+)_[^/]+/images/(\w+)/D\d+_I_\w+?_(\d+)\.jpg$", str(p))
        if not m:
            continue
        dev, ver, num = m.groups()
        try:
            with Image.open(p) as im:
                im.load()
        except Exception:
            bad.append(str(p)); continue
        rows.append(dict(path=str(p), label=0, split="test", source=ver, device=dev, scene=f"{dev}_{'flat' if ver == 'flat' else 'nat'}_{num}"))
    # keep only natural scenes present in all four versions, so version comparisons are paired
    nat = {}
    for r in rows:
        if r["source"] != "flat":
            nat.setdefault(r["scene"], set()).add(r["source"])
    full = {s for s, v in nat.items() if v == {"nat", "natFBH", "natFBL", "natWA"}}
    rows = [r for r in rows if r["source"] == "flat" or r["scene"] in full]
    d = ROOT / "runs" / "e5"; d.mkdir(parents=True, exist_ok=True)
    with open(d / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source", "device", "scene"]); w.writeheader(); w.writerows(rows)
    np.save(d / "idx_all.npy", np.arange(len(rows)))
    from collections import Counter
    print(len(rows), "rows;", dict(Counter(r["source"] for r in rows)), f"; {len(bad)} unreadable", bad[:3])


if __name__ == "__main__":
    main()
