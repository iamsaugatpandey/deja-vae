"""AEROBLADE baseline (Ricker, Lukovnikov & Fischer, CVPR 2024): training-free. An image x is scored by
its autoencoder reconstruction distance  d(x) = min_k LPIPS_2(x, R_k(x)), where R_k = encode -> decode
with autoencoder k and LPIPS_2 is the 2nd VGG layer of LPIPS (the paper's best layer), averaged over
the spatial map. LOWER distance = more likely generated. Score stored as -d (higher = generated).

Autoencoders as in the paper: SD1 (kl-f8; the SD v1.1-1.5 VAE is one checkpoint), SD2 (the SD 2.x
VAE) and Kandinsky 2.1 (MoVQ).
Input: the SAME native-resolution view as Déjà VAE (centre crop <= 256 px on the 8-px grid, then
the set's test JPEG re-save), so the comparison isolates the scoring rule, not the crop.
Note: the paper used whole images; a 256 crop is a fair-compute choice and is stated in the paper.

Cost: three autoencoders per image, so scored on a FIXED stratified subset of the eval rows
(seed 13): GenImage 1000 calib reals, 1000 test reals (stratified by source), 250 fakes per generator;
Chameleon 1000 calib reals, 1000 test reals, 2000 fakes. Degraded copies of the test reals as in
`dejavae.py pairs`. `report` scores EVERY method on the same subset rows (paired tests).

    anchor/bin/python aeroblade.py run --set genimage|cham
    DV_WORK=runs/dv2 DV_FEAT=mid anchor/bin/python aeroblade.py report --set genimage|cham
"""
import os, sys, json, time, argparse
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv

AES = {   # name -> list of specs tried in order (first one that loads wins; mirrors share weights)
    "sd1": ["stable-diffusion-v1-5/stable-diffusion-v1-5:vae", "CompVis/stable-diffusion-v1-1:vae"],
    "sd2": ["stabilityai/stable-diffusion-2-base:vae", "sd-research/stable-diffusion-2-1-base:vae"],
    "kd21": ["kandinsky-community/kandinsky-2-1:movq"],
}
N = {"genimage": dict(calib=1000, real=1000, fake_per_src=250), "cham": dict(calib=1000, real=1000, fake_per_src=2000),
     "cham_m": dict(calib=1000, real=1000, fake_per_src=2000),
     "sb": dict(calib=1000, real=1800, fake_per_src=200)}       # E5: 200 matched reals + 200 fakes per generator


def out_dir(s):
    d = Path(dv.SETS[s][1]) / "aeroblade"; d.mkdir(parents=True, exist_ok=True); return d


def subset(s):
    rows, idx = dv.rows_of(s)
    rng = np.random.default_rng(13); n = N[s]
    lab, split, src = ap.split_idx(rows); ix = np.array(idx)
    pick = list(rng.choice(ix[(split[ix] == "calib") & (lab[ix] == 0)], n["calib"], replace=False))
    tr = ix[(split[ix] == "test") & (lab[ix] == 0)]; srcs = sorted(set(src[tr]))
    per = n["real"] // len(srcs)
    for g in srcs:
        c = tr[src[tr] == g]; pick += list(rng.choice(c, min(per, len(c)), replace=False))
    for g in sorted(set(src[ix][lab[ix] == 1])):
        c = ix[(split[ix] == "test") & (lab[ix] == 1) & (src[ix] == g)]
        pick += list(rng.choice(c, min(n["fake_per_src"], len(c)), replace=False))
    return rows, sorted(int(i) for i in pick)


def cmd_run(a):
    import torch, lpips
    from PIL import ImageFilter
    rows, sub = subset(a.set); O = out_dir(a.set)
    np.save(O / "idx.npy", np.array(sub))
    tq = dv.SETS[a.set][2]
    J = (lambda im: ap.jpeg(im, tq)) if tq else (lambda im: im)
    Nall = len(rows)
    rng = np.random.default_rng(0)                      # degraded copies: same draws as dejavae pairs / features
    q = np.where(rng.random(Nall) < 1.0, rng.integers(70, 101, Nall), 0)
    sig = rng.uniform(0.5, 1.5, Nall)
    dev = ap.pick_device()
    enc = ap.Encoders(dev, need_vae=False, norm="simlbr"); enc.dino = None
    if dev.type == "mps":
        torch.mps.empty_cache()
    lp = lpips.LPIPS(net="vgg", lpips=True, spatial=True, verbose=False).to(dev).eval()
    deg_ids = [i for i in sub if rows[i]["label"] == 0 and rows[i]["split"] == "test"]
    print(f"aeroblade {a.set}: {len(sub)} rows + {len(deg_ids)} degraded", flush=True)

    def lpips2(A, B):
        t = lambda ims: enc.to_tensor(ims) * 2 - 1
        with torch.no_grad():
            _, per = lp(t(A), t(B), retPerLayer=True, normalize=False)
        return per[1].mean((1, 2, 3)).float().cpu().numpy()     # 2nd VGG layer, spatial mean

    def dists(ims):
        out = np.zeros(len(ims), np.float32); by = {}
        for i, im in enumerate(ims):
            by.setdefault(im.size, []).append(i)
        for ids in by.values():
            for s in range(0, len(ids), 8):
                b = ids[s:s + 8]; X = [ims[i] for i in b]
                out[b] = lpips2(X, enc.reconstruct(X, 4))
        return out

    for name, specs in AES.items():
        f = O / f"d_{name}.npz"
        if f.exists():
            print(f"  {name} already done", flush=True); continue
        for sp in specs:
            try:
                enc.vae = ap.load_autoencoder(sp, dev); print(f"  {name}: {sp}", flush=True); break
            except Exception as e:
                print(f"  {name}: {sp} failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
        else:
            raise SystemExit(f"no checkpoint for {name}")
        t0 = time.time(); d = np.zeros(len(sub), np.float32); dg = np.zeros(len(deg_ids), np.float32)
        CH = 64
        for s in range(0, len(sub), CH):
            b = sub[s:s + CH]
            d[s:s + len(b)] = dists([J(dv.view(rows[i]["path"])) for i in b])
            if (s // CH) % 10 == 0:
                el = time.time() - t0; done = s + len(b)
                print(f"  {name} {done}/{len(sub)}  {done / el:.1f} img/s  eta {(len(sub) - done + len(deg_ids)) / (done / el) / 60:.0f} min", flush=True)
        for s in range(0, len(deg_ids), CH):
            b = deg_ids[s:s + CH]
            dg[s:s + len(b)] = dists([J(dv.view(rows[i]["path"]).filter(ImageFilter.GaussianBlur(float(sig[i])))) for i in b])
        np.savez(f, d=d, deg=dg, deg_ids=np.array(deg_ids))
        enc.vae = None
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(f"  {name} done in {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"AEROBLADE {a.set} DONE", flush=True)


def cmd_report(a):
    """Every method on the SAME subset rows: evaluate() + paired bootstrap vs AEROBLADE."""
    import pandas as pd
    rows, idx = dv.rows_of(a.set); O = out_dir(a.set)
    sub = np.load(O / "idx.npy"); Nall = len(rows)
    D = {n: np.load(O / f"d_{n}.npz") for n in AES if (O / f"d_{n}.npz").exists()}
    runs = {}

    def full(vals, ids):
        v = np.zeros(Nall, np.float32); v[ids] = vals; return v

    deg_ids = next(iter(D.values()))["deg_ids"]
    variants = {f"aeroblade-{n}": [n] for n in D}
    if len(D) > 1:
        variants["aeroblade"] = list(D)                  # the paper's rule: min over the autoencoders
    for m, names in variants.items():
        sc = -np.min([D[n]["d"] for n in names], 0); dg = -np.min([D[n]["deg"] for n in names], 0)
        runs[m] = {s: (full(sc, sub), full(dg, deg_ids)) for s in (0, 1, 2)}
    ref = Path(dv.SETS[a.set][1])
    for m in ("simlbr_official", "lbr_exact", "rigid-dinov2", "rigid-dinov2~norm", "rigid-dinov3", "recon", "sup"):
        if (ref / f"scores_{m}_seed0.npy").exists():
            runs[m] = {s: (np.load(ref / f"scores_{m}_seed{s}.npy"), np.load(ref / f"degscores_{m}_seed{s}.npy")) for s in (0, 1, 2)}
    for f in sorted(O.glob("scores_rigid*_seed0.npy")):        # subset-only RIGID sanity variants (raw / q95, noise space)
        m = f.stem[len("scores_"):-len("_seed0")]
        runs[m] = {s: (np.load(O / f"scores_{m}_seed{s}.npy"), np.load(O / f"degscores_{m}_seed{s}.npy")) for s in (0, 1, 2)}
    dvr = dv.scores_for(a.set, rows, list(sub))
    for h in ("dv", "dv-single", "dv-tf"):
        if h in dvr:
            runs[h] = dvr[h]
    srows = [rows[i] for i in sub]
    runs = {m: {s: (sc[sub], dg[sub]) for s, (sc, dg) in b.items()} for m, b in runs.items()}
    for v in ("dv", "aeroblade"):
        if v in runs and "simlbr_official" in runs:
            for s in (0, 1, 2):
                (d, dg), (l, lg) = runs[v][s], runs["simlbr_official"][s]
                f = dv.zfuse(d, l, srows)
                runs.setdefault(f"{v}+fuse-simlbr_official", {})[s] = (f(d, l), f(dg, lg))
    summ, per = [], []
    for m, b in runs.items():
        for s, (sc, dg) in b.items():
            out, sm = ap.evaluate(sc, dg, srows); sm.update(method=m, seed=s); summ.append(sm)
            per += [dict(method=m, seed=s, source=o["source"], tpr5=o["tpr@5%"], auroc=o["auroc"]) for o in out]
    cols = ["mean_tpr@5%", "worst_tpr@5%", "mean_tpr@1%", "fpr_test_reals@5%", "fpr_degraded_reals@5%"]
    t = pd.DataFrame(summ).groupby("method")[cols].mean().round(1).sort_values("mean_tpr@5%", ascending=False)
    lab, split, _ = ap.split_idx(srows)
    lines = [f"AEROBLADE + all methods on the fixed stratified subset of {a.set} ({time.strftime('%Y-%m-%d %H:%M')})",
             f"n_calib_real={int(((split == 'calib') & (lab == 0)).sum())}  n_test_real={int(((split == 'test') & (lab == 0)).sum())}  "
             f"n_test_fake={int(((split == 'test') & (lab == 1)).sum())}; deterministic methods copied per seed", "", t.to_string(), ""]
    pt = pd.DataFrame(per).groupby(["method", "source"])["tpr5"].mean().unstack().round(1)
    if pt.shape[1] > 1:
        lines += ["TPR@5% per generator:", pt.to_string(), ""]
    lines.append("Paired bootstrap (B=1000; resample calib reals + test fakes; seed-averaged), TPR@5%:")
    for x, y in (("dv-single", "aeroblade"), ("dv", "aeroblade"), ("dv-tf", "aeroblade"), ("aeroblade", "rigid-dinov2"), ("aeroblade", "rigid-dinov2~norm"), ("dv-single", "rigid-dinov2~norm"),
                 ("dv+fuse-simlbr_official", "aeroblade+fuse-simlbr_official")):
        if x in runs and y in runs:
            r = ap.paired_bootstrap([runs[x][s][0] for s in (0, 1, 2)], [runs[y][s][0] for s in (0, 1, 2)], srows, eps=0.05, B=1000)
            (mm, ml, mh), (wm, wl, wh) = r["mean"], r["worst"]
            lines.append(f"  {x:26s} - {y:30s} mean {mm:+5.1f} [{ml:+.1f},{mh:+.1f}]  worst {wm:+5.1f} [{wl:+.1f},{wh:+.1f}]")
    txt = "\n".join(lines); print(txt); (O / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("run", "report"):
        s = sub.add_parser(c); s.add_argument("--set", choices=["genimage", "cham", "cham_m", "sb"], required=True)
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)
