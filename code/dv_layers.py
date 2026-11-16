"""Déjà VAE diagnostic: WHERE in DINOv3 does the autoencoder fixed point show up?

The pilot's relational head compared final-layer CLS vectors, phi(v) vs phi(R(v)), and did not
beat a single-image head. Hypothesis: the final CLS is built to ignore the low-level changes R
makes, and the p2 JPEG q95 re-save knocks generated images off the fixed point.

Training-free check, no learning: for each image view v (same native-res crop as dejavae.py),
compute R(v) with the SD VAE and measure how far v moves, at every 2nd DINOv3 layer:
    cls   cosine distance of CLS tokens
    pmean mean over the 16x16 patch grid of per-patch cosine distance (R keeps geometry)
    p90   90th percentile of per-patch distance
    psnr  pixel baseline (lower PSNR = moved more)
Score = -distance (moved less = more likely generated). AUROC real vs each generator, with and
without the JPEG q95 re-save. Also: simulated fakes R(x) and blur-matched reals (the training rows).

    anchor/bin/python dv_layers.py run      # ~20-30 min on the Mac, resumable per group
    anchor/bin/python dv_layers.py report
"""
import sys, io, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
from dejavae import view, to256

OUT = Path("runs/dv/layers"); OUT.mkdir(parents=True, exist_ok=True)
LAYERS = list(range(2, 25, 2))
N_REAL, N_GEN, N_CHAM = 200, 40, 150


def jpeg95(im):
    from PIL import Image
    b = io.BytesIO(); im.save(b, "JPEG", quality=95); b.seek(0)
    return Image.open(b).convert("RGB")


class Probe:
    def __init__(self):
        self.enc = ap.Encoders(ap.pick_device(), norm="simlbr")
        cfg = self.enc.dino.config
        self.skip = 1 + int(getattr(cfg, "num_register_tokens", 0))

    def tokens(self, ims):
        t = self.enc.torch
        with t.no_grad():
            x = (self.enc.to_tensor([to256(im) for im in ims]) - self.enc.mean) / self.enc.std
            hs = self.enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
            return {L: hs[L].float() for L in LAYERS}

    def measure(self, A, B):
        """distance measures between views A and their partners B (same sizes, pairwise)."""
        t = self.enc.torch
        res = {}
        for s in range(0, len(A), 8):
            ta, tb = self.tokens(A[s:s + 8]), self.tokens(B[s:s + 8])
            for L in LAYERS:
                cos = t.nn.functional.cosine_similarity(ta[L], tb[L], dim=-1)   # [n, tokens]
                d = (1 - cos).cpu().numpy()
                res.setdefault(f"cls@{L}", []).extend(d[:, 0])
                res.setdefault(f"pmean@{L}", []).extend(d[:, self.skip:].mean(1))
                res.setdefault(f"p90@{L}", []).extend(np.percentile(d[:, self.skip:], 90, axis=1))
        res["psnr"] = [-ap.psnr(a, b) for a, b in zip(A, B)]   # stored as a distance: higher = moved more
        return {k: np.asarray(v, np.float32) for k, v in res.items()}

    def recon(self, ims):
        out = [None] * len(ims); by = {}
        for i, im in enumerate(ims):
            by.setdefault(im.size, []).append(i)
        for idx in by.values():
            for i, r in zip(idx, self.enc.reconstruct([ims[i] for i in idx], 4)):
                out[i] = r
        return out


def groups():
    rng = np.random.default_rng(7)
    g = {}
    rows = ap.read_manifest(Path("runs/p2/manifest.csv"))
    ev = [r for r in rows if r["split"] != "train"]
    reals = [r["path"] for r in ev if r["label"] == 0]
    g["gi:real"] = [reals[i] for i in rng.choice(len(reals), N_REAL, replace=False)]
    for src in sorted({r["source"] for r in ev if r["label"] == 1}):
        c = [r["path"] for r in ev if r["label"] == 1 and r["source"] == src]
        g[f"gi:{src}"] = [c[i] for i in rng.choice(len(c), min(N_GEN, len(c)), replace=False)]
    cr = ap.read_manifest(Path("runs/cham/manifest.csv"))
    ce = [r for r in cr if r["split"] != "train"]
    for lab, nm in ((0, "real"), (1, "fake")):
        c = [r["path"] for r in ce if r["label"] == lab]
        g[f"cham:{nm}"] = [c[i] for i in rng.choice(len(c), min(N_CHAM, len(c)), replace=False)]
    return g


def cmd_run():
    P = Probe(); G = groups()
    json.dump(G, open(OUT / "groups.json", "w"), indent=0)
    for name, paths in G.items():
        conds = ["raw", "jpeg"] if name.startswith("gi:") else ["raw"]
        for cond in conds:
            f = OUT / f"{name.replace(':', '_')}_{cond}.npz"
            if f.exists():
                continue
            t0 = time.time()
            ims = [view(p) for p in paths]
            if cond == "jpeg":
                ims = [jpeg95(im) for im in ims]
            R = P.recon(ims)
            out = {f"x/{k}": v for k, v in P.measure(ims, R).items()}
            if name == "gi:real":   # the training rows: simulated fake R(x) and blur-matched real
                RR = P.recon(R)
                out.update({f"sim/{k}": v for k, v in P.measure(R, RR).items()})
                D = [ap.blur_to_psnr(a, ap.psnr(a, b)) for a, b in zip(ims, R)]
                out.update({f"blur/{k}": v for k, v in P.measure(D, P.recon(D)).items()})
            np.savez(f, **out)
            print(f"{name:32s} {cond:4s} n={len(ims)}  {time.time() - t0:5.0f}s", flush=True)
    print("LAYERS RUN DONE", flush=True)


def auroc(pos, neg):
    """P(score_pos > score_neg); score = -distance, pos = generated."""
    from scipy.stats import rankdata
    s = np.concatenate([-pos, -neg]); r = rankdata(s)
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_report():
    import pandas as pd
    L = lambda n, c: dict(np.load(OUT / f"{n}_{c}.npz"))
    lines = ["Déjà VAE layer diagnostic: AUROC, generated vs real, score = -(distance between v and R(v))",
             "0.5 = no separation; >0.5 = generated images move LESS (the fixed point); <0.5 = they move MORE", ""]
    keys = [k[2:] for k in L("gi_real", "raw") if k.startswith("x/")]
    order = ["psnr"] + [f"{m}@{l}" for m in ("cls", "pmean", "p90") for l in LAYERS]
    keys = [k for k in order if k in keys]
    gens = sorted(f.stem[3:-4] for f in OUT.glob("gi_*_raw.npz") if f.stem != "gi_real_raw")
    for cond in ("raw", "jpeg"):
        real = L("gi_real", cond)
        tab = {}
        for k in keys:
            row = {g: auroc(L(f"gi_{g}", cond)[f"x/{k}"], real[f"x/{k}"]) for g in gens}
            row["MEAN"] = np.mean([row[g] for g in gens])
            row["simR(x)"] = auroc(real[f"sim/{k}"], real[f"x/{k}"])
            row["blur*"] = auroc(real[f"blur/{k}"], real[f"x/{k}"])
            if cond == "raw":
                c = L("cham_real", "raw"), L("cham_fake", "raw")
                row["CHAM"] = auroc(c[1][f"x/{k}"], c[0][f"x/{k}"])
            tab[k] = row
        df = pd.DataFrame(tab).T.round(2)
        df.columns = [c.replace("stable_diffusion_v_1_", "SD1.") for c in df.columns]
        lines += [f"== GenImage, {'no re-save' if cond == 'raw' else 'JPEG q95 re-save (the p2 protocol)'}"
                  + ("; CHAM = Chameleon (no re-save)" if cond == "raw" else ""),
                  "   simR(x) = simulated fake R(x) vs real (what training sees); blur* = blur-matched real vs real (want ~0.5)",
                  df.to_string(), ""]
    txt = "\n".join(lines); print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
