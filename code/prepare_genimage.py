#!/usr/bin/env python3
"""Download GenImage from the sharded Hugging Face copy (nebula/GenImage-arrow) and write each image
to disk with its ORIGINAL bytes and file name (no re-encoding: GenImage's JPEG-real / PNG-fake
format difference is a known shortcut, so we keep the files exactly as released).

Layout produced:  data/genimage/<generator>/<train|test>/<nature|ai>/<file>
Each arrow shard is deleted after export, so peak extra disk is ~1 shard per worker.

    anchor/bin/python prepare_genimage.py                 # defaults below
    anchor/bin/python prepare_genimage.py --status        # what is done
"""
import argparse, json, os, sys, time
from pathlib import Path

REPO = "nebula/GenImage-arrow"
GENS = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"]
# SD1.4 train: rows are ordered real (0..161999) then fake. Shards spread over each half -> ~10k each.
TRAIN_SHARDS = {"stable_diffusion_v_1_4": [0, 15, 30, 45, 60, 75, 100, 115, 130, 145, 160, 175]}

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "genimage"
TMP = HERE / "data" / "genimage_arrow"
DONE = OUT / "_exported.json"


def manifest():
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(REPO, "manifest.json", repo_type="dataset")))


def jobs(m):
    out = []
    for g in GENS:
        for s in m["bundles"]["test"][g]["shards"]:
            out.append(s)
    for g, idx in TRAIN_SHARDS.items():
        sh = m["bundles"]["train"][g]["shards"]
        out += [sh[i] for i in idx]
    return out


def export(shard_path):
    from datasets import Dataset
    ds = Dataset.from_file(str(shard_path))
    n = 0
    for r in ds:
        p = Path(r["image_path"])                 # e.g. BigGAN/test/nature/ILSVRC2012_val_00000009.JPEG
        dst = OUT / p
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(r["image"]); tmp.rename(dst)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    m = manifest(); todo = jobs(m)
    done = set(json.load(open(DONE))) if DONE.exists() else set()
    tot_b = sum(s["size_bytes"] for s in todo); done_b = sum(s["size_bytes"] for s in todo if s["path"] in done)
    if a.status:
        print(f"GenImage shards exported: {len(done & {s['path'] for s in todo})}/{len(todo)}  "
              f"({done_b / 1e9:.1f}/{tot_b / 1e9:.1f} GB)"); return
    t0, got = time.time(), 0
    for k, s in enumerate(todo):
        if s["path"] in done:
            continue
        local = Path(hf_hub_download(REPO, s["path"], repo_type="dataset", local_dir=TMP))
        n = export(local)
        local.unlink(missing_ok=True)
        done.add(s["path"]); json.dump(sorted(done), open(DONE, "w"))
        got += s["size_bytes"]; done_b += s["size_bytes"]
        rate = got / max(time.time() - t0, 1e-6)
        print(f"[{k + 1}/{len(todo)}] {s['path']}  {n} images  "
              f"{done_b / 1e9:.1f}/{tot_b / 1e9:.1f} GB  ~{(tot_b - done_b) / rate / 60:.0f} min left", flush=True)
    print("ALL SHARDS EXPORTED")


if __name__ == "__main__":
    main()
