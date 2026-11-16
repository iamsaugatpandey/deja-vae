"""Download RAISE-1k TIFFs listed in the CSV the user obtains after registering at
https://loki.disi.unitn.it/RAISE/confirm.php?package=1k (licence: non-commercial research/education; cite
Dang-Nguyen et al., RAISE, ACM MMSys 2015). Only the 1,000 ids that Synthbuster uses are fetched.
Files: ~/anchor_data/raise/tiff/<id>.TIF  (~35-50 MB each). Resumable; 4 parallel connections.
    anchor/bin/python -I raise_download.py ~/anchor_data/raise/RAISE_1k.csv
"""
import sys, csv, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SB = Path.home() / "anchor_data" / "synthbuster" / "synthbuster" / "stable-diffusion-1-4"
DST = Path.home() / "anchor_data" / "raise" / "tiff"


def main(csv_path):
    want = {p.stem for p in SB.glob("*.png")}
    rows = list(csv.DictReader(open(Path(csv_path).expanduser(), newline="", encoding="utf-8-sig")))
    col = next(c for c in rows[0] if c.strip().upper() == "TIFF")
    jobs = []
    for r in rows:
        url = r[col].strip(); i = Path(url).stem
        if i in want:
            jobs.append((i, url))
    print(f"{len(jobs)} of {len(want)} Synthbuster ids found in the CSV"); DST.mkdir(parents=True, exist_ok=True)

    def get(job):
        i, url = job; out = DST / f"{i}.TIF"
        if out.exists() and out.stat().st_size > 1_000_000:
            return 0
        part = out.with_suffix(".part")
        c = subprocess.call(["curl", "-sfL", "--retry", "5", "-A", "research-client/1.0", "-o", str(part), url])
        if c == 0:
            part.rename(out)
        return c

    codes = []
    with ThreadPoolExecutor(4) as ex:
        for k, c in enumerate(ex.map(get, jobs), 1):
            codes.append(c)
            if k % 25 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    bad = [j[0] for j, c in zip(jobs, codes) if c != 0]
    print(f"done: {len(jobs) - len(bad)} ok, {len(bad)} failed {bad[:5]}")


if __name__ == "__main__":
    main(sys.argv[1])
