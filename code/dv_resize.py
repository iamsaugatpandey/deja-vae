"""Why does zero-fake Déjà VAE v2 beat SimLBR on GenImage but collapse on Chameleon?
Hypothesis: in-the-wild images are resized / recompressed after generation, which breaks the
autoencoder fixed point (pilot: resizing before R destroys it).

  A. sizes   Chameleon: TPR@5% of the v2 heads by image size and by whether both sides are multiples
             of 64 (raw generator output usually is; resized images usually are not). No GPU.
  B. post    GenImage SD1.4 / wukong / Midjourney fakes + reals, post-processed (resize x0.5, x0.75,
             x1.5, JPEG q75, x0.75 + q75), then the standard pipeline (native crop, JPEG q95). Scored
             with v2 heads (dv, dv-single, dv-nocf) and SimLBR (lbr_exact ours, released head).
             TPR@5% with the threshold re-set on reals processed the same way, plus the false-alarm
             rate of the unprocessed-real threshold on processed reals.

    DV_FEAT=mid DV_WORK=runs/dv2 anchor/bin/python dv_resize.py sizes|post|report
"""
import os, sys, io, json, time
os.environ.setdefault("DV_FEAT", "mid"); os.environ.setdefault("DV_WORK", "runs/dv2")
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv

OUT = Path(os.environ["DV_WORK"]) / "resize"; OUT.mkdir(parents=True, exist_ok=True)
HEADS = ["dv", "dv-single", "dv-nocf"]
CONDS = {"none": (1.0, None), "x0.75": (0.75, None), "x0.5": (0.5, None), "x1.5": (1.5, None),
         "q75": (1.0, 75), "x0.75+q75": (0.75, 75)}
GENS = ["stable_diffusion_v_1_4", "wukong", "Midjourney"]
N_REAL, N_FAKE = 300, 150


def thr_tpr(real, fake, eps=0.05):
    t = np.quantile(real, 1 - eps)
    return 100 * float((fake > t).mean()), t


# ----------------------------------------------------------------------------- A. Chameleon sizes
def cmd_sizes():
    from PIL import Image
    import pandas as pd
    rows, idx = dv.rows_of("cham")
    runs = dv.scores_for("cham", rows, idx)
    lab, split, _ = ap.split_idx(rows)
    sz = {}
    for i in idx:
        with Image.open(rows[i]["path"]) as im:
            sz[i] = (im.size, (im.format or "?"))
    ix = np.array(idx)
    W = np.array([sz[i][0][0] for i in ix]); H = np.array([sz[i][0][1] for i in ix])
    fmt = np.array([sz[i][1] for i in ix])
    cal = (split[ix] == "calib") & (lab[ix] == 0); fk = (split[ix] == "test") & (lab[ix] == 1)
    tr = (split[ix] == "test") & (lab[ix] == 0)
    m64 = (W % 64 == 0) & (H % 64 == 0); m8 = (W % 8 == 0) & (H % 8 == 0)
    short = np.minimum(W, H)
    groups = {"all": np.ones(len(ix), bool), "both sides %64==0": m64, "%8 but not %64": m8 & ~m64,
              "not %8": ~m8, "short<512": short < 512, "512-767": (short >= 512) & (short < 768),
              "768-1023": (short >= 768) & (short < 1024), "1024+": short >= 1024}
    for f in sorted(set(fmt)):
        groups[f"format {f}"] = fmt == f
    lines = ["A. Chameleon: TPR@5% (threshold from all calib reals) by image-size group; FPR = test reals in that group", ""]
    tab = []
    for g, m in groups.items():
        row = dict(group=g, n_fake=int((fk & m).sum()), n_real=int((tr & m).sum()))
        for h in HEADS + ["dv-tf"]:
            tp, fp = [], []
            for s in (0, 1, 2):
                sc = runs[h][s][0][ix]
                t = np.quantile(sc[cal], 0.95)
                tp.append(100 * (sc[fk & m] > t).mean() if (fk & m).any() else np.nan)
                fp.append(100 * (sc[tr & m] > t).mean() if (tr & m).any() else np.nan)
            row[f"{h} TPR"] = round(float(np.mean(tp)), 1); row[f"{h} FPR"] = round(float(np.mean(fp)), 1)
        tab.append(row)
    lines.append(pd.DataFrame(tab).set_index("group").to_string())
    txt = "\n".join(lines); print(txt); (OUT / "sizes.txt").write_text(txt + "\n")


# ----------------------------------------------------------------------------- B. post-processing
def view_im(im):
    w, h = im.size
    cw, ch = min(256, w // 8 * 8), min(256, h // 8 * 8)
    l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return im.crop((l, t, l + cw, t + ch))


def squash256(im):
    import torch
    from torchvision.transforms import v2
    from PIL import Image
    t = v2.Resize((256, 256), antialias=True)(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def post(im, s, q):
    from PIL import Image
    if s != 1.0:
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC)
    return ap.jpeg(im, q) if q else im


def cmd_post():
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    rows = ap.read_manifest(Path("runs/p2/manifest.csv"))
    rng = np.random.default_rng(11)
    ev = [i for i, r in enumerate(rows) if r["split"] != "train"]
    reals = [i for i in ev if rows[i]["label"] == 0]
    sel = {"real": rng.choice(reals, N_REAL, replace=False).tolist()}
    for g in GENS:
        c = [i for i in ev if rows[i]["label"] == 1 and rows[i]["source"] == g]
        sel[g] = rng.choice(c, N_FAKE, replace=False).tolist()
    json.dump({k: [rows[i]["path"] for i in v] for k, v in sel.items()}, open(OUT / "selection.json", "w"))
    enc = ap.Encoders(ap.pick_device(), norm="simlbr")
    P = dv.Pairs(enc)
    dvh = {h: [ap.head_from_state(torch.load(f"{os.environ['DV_WORK']}/head_{h}_seed{s}.pt", map_location="cpu")) for s in (0, 1, 2)] for h in HEADS}
    lbr = [ap.head_from_state(torch.load(f"runs/p4s/head_lbr_exact_seed{s}.pt", map_location="cpu")) for s in (0, 1, 2)]
    off = ap.head_from_state(load_file(hf_hub_download("MVRL/SimLBR", "genimage/model.safetensors")))
    mode = {h: dv.VARIANTS[h][0] for h in HEADS}
    for cname, (s, q) in CONDS.items():
        f = OUT / f"post_{cname}.npz"
        if f.exists():
            continue
        t0 = time.time(); res = {}
        for grp, ids in sel.items():
            sc = {k: [] for k in [f"{h}/{k}" for h in HEADS for k in (0, 1, 2)] + [f"lbr_exact/{k}" for k in (0, 1, 2)] + ["simlbr_official/0"]}
            for b in range(0, len(ids), 16):
                ims = [post(Image.open(rows[i]["path"]).convert("RGB"), s, q) for i in ids[b:b + 16]]
                v = [ap.jpeg(view_im(im), 95) for im in ims]
                a_, b_ = P.phi(v), P.phi(P.recon(v))
                sq = enc.cls([ap.jpeg(squash256(im), 95) for im in ims])
                with torch.no_grad():
                    for h in HEADS:
                        x = torch.from_numpy(dv.rel(a_, b_, mode[h]))
                        for k in (0, 1, 2):
                            sc[f"{h}/{k}"].extend(dvh[h][k](x).squeeze(1).numpy())
                    xs = torch.from_numpy(sq.astype(np.float32))
                    for k in (0, 1, 2):
                        sc[f"lbr_exact/{k}"].extend(lbr[k](xs).squeeze(1).numpy())
                    sc["simlbr_official/0"].extend(off(xs).squeeze(1).numpy())
            for k, v_ in sc.items():
                res[f"{grp}|{k}"] = np.asarray(v_, np.float32)
        np.savez(f, **res)
        print(f"post {cname:10s} done in {time.time() - t0:5.0f}s", flush=True)
    print("POST DONE", flush=True)
    cmd_report()


def cmd_report():
    import pandas as pd
    D = {c: dict(np.load(OUT / f"post_{c}.npz")) for c in CONDS if (OUT / f"post_{c}.npz").exists()}
    methods = HEADS + ["lbr_exact", "simlbr_official"]
    tab = []
    for m in methods:
        seeds = (0,) if m == "simlbr_official" else (0, 1, 2)
        for c, d in D.items():
            row = dict(method=m, cond=c)
            per = {g: [] for g in GENS}; fpr = []
            for k in seeds:
                real = d[f"real|{m}/{k}"]
                for g in GENS:
                    per[g].append(thr_tpr(real, d[f"{g}|{m}/{k}"])[0])
                t_fixed = np.quantile(D["none"][f"real|{m}/{k}"], 0.95)
                fpr.append(100 * float((real > t_fixed).mean()))
            for g in GENS:
                row[g.replace("stable_diffusion_v_1_4", "SD1.4")] = round(float(np.mean(per[g])), 1)
            row["MEAN"] = round(float(np.mean([np.mean(per[g]) for g in GENS])), 1)
            row["real FPR@fixed thr"] = round(float(np.mean(fpr)), 1)
            tab.append(row)
    df = pd.DataFrame(tab).set_index(["method", "cond"])
    lines = ["B. GenImage fakes + reals post-processed BEFORE the standard pipeline; TPR@5% with the threshold",
             "   re-set on reals processed the same way (n_real=300, n_fake=150 per generator); last column: false-alarm",
             "   rate on processed reals if the unprocessed-real threshold is kept", "", df.to_string()]
    txt = "\n".join(lines); print(txt); (OUT / "post.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"sizes": cmd_sizes, "post": cmd_post, "report": cmd_report}[sys.argv[1]]()
