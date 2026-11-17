"""Review: reproduce the 'blindness' of final-layer probes (Zhou et al., 2026) in our own setup.
1,000 GenImage test reals (seed 5, both sides <= 640 px; larger copies swap on the 16 GB laptop) are copied ONCE at full resolution with the ft-MSE autoencoder
(= the SD 2 autoencoder), as a real photograph passed through a VAE would be, and then go through each detector's own
test pipeline: SimLBR / supervised / reconstruction probe: squash to 256, JPEG q95, final-layer class token;
Deja VAE-I: native 256 centre crop, JPEG q95, layers 8-18 (runs/dv2) or final layer (runs/dv).
Thresholds are the paper's (5% FPR on the calibration reals). Reports the share of copies flagged as generated, and the
share of the same originals flagged (control, should be near 5%). Mean over 3 seeds (SimLBR released: one head).
    anchor/bin/python review_copies.py   -> runs/review_copies.txt
"""
import os, sys
os.environ["DV_FEAT"] = "mid"; os.environ["DV_WORK"] = "runs/dv2"
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv


def squash(im):
    import torch
    from torchvision.transforms import v2
    from PIL import Image
    t = v2.Resize((256, 256), antialias=True)(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def crop8(im):
    w, h = im.size; return im.crop((0, 0, w // 8 * 8, h // 8 * 8))


def vcrop(im):
    w, h = im.size; cw, ch = min(256, w // 8 * 8), min(256, h // 8 * 8)
    l, t = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return im.crop((l, t, l + cw, t + ch))


def main():
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    rows, idx = dv.rows_of("genimage"); lab, split, src = ap.split_idx(rows)
    cal = (split == "calib") & (lab == 0)
    cand = [i for i in np.where((split == "test") & (lab == 0))[0]]
    rng = np.random.default_rng(5); rng.shuffle(cand); pick = []
    for i in cand:
        w, h = Image.open(rows[i]["path"]).size
        if max(w, h) <= 640:
            pick.append(i)
        if len(pick) == 1000:
            break
    enc = ap.Encoders(ap.pick_device(), norm="simlbr", vae=dv.AE)
    feats = {k: [] for k in ("sq_orig", "sq_copy", "mid_orig", "mid_copy", "fin_orig", "fin_copy")}
    for s in range(0, len(pick), 8):
        ims = [crop8(Image.open(rows[i]["path"]).convert("RGB")) for i in pick[s:s + 8]]
        cps = [enc.reconstruct([im], 1)[0] for im in ims]          # full-resolution copies, one size at a time
        for tag, group in (("orig", ims), ("copy", cps)):
            sq = [ap.jpeg(squash(im), 95) for im in group]
            feats[f"sq_{tag}"].append(enc.cls(sq))
            v = [ap.jpeg(vcrop(im), 95) for im in group]
            with torch.no_grad():
                x = (enc.to_tensor([dv.to256(im) for im in v]) - enc.mean) / enc.std
                hs = enc.dino(pixel_values=x, output_hidden_states=True)
                feats[f"mid_{tag}"].append(torch.cat([hs.hidden_states[L][:, 0] for L in dv.MID], 1).float().cpu().numpy())
                feats[f"fin_{tag}"].append(hs.last_hidden_state[:, 0].float().cpu().numpy())
        if s % 80 == 0:
            print(f"{s}/{len(pick)}", flush=True)
    F = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    L = [__doc__.strip().splitlines()[0], f"n = {len(pick)} test reals; share flagged at the paper's 5% threshold (%)", ""]
    def report(name, heads, ref_scores, key):
        o, c = [], []
        for h, ref in zip(heads, ref_scores):
            tau = ap.conformal_tau(ref[cal], 0.05)
            with torch.no_grad():
                o.append((h(torch.from_numpy(F[f"{key}_orig"])).squeeze(1).numpy() > tau).mean() * 100)
                c.append((h(torch.from_numpy(F[f"{key}_copy"])).squeeze(1).numpy() > tau).mean() * 100)
        L.append(f"{name:42s} originals {np.mean(o):5.1f}   copies {np.mean(c):5.1f}"); print(L[-1], flush=True)
    ld = lambda p: ap.head_from_state(torch.load(p, map_location="cpu"))
    sc = lambda m: [np.load(f"runs/p4s/scores_{m}_seed{s}.npy") for s in (0, 1, 2)]
    report("SimLBR (released; trained on SD1.4 images)", [ap.head_from_state(load_file(hf_hub_download("MVRL/SimLBR", "genimage/model.safetensors")))],
           [np.load("runs/p4s/scores_simlbr_official_seed0.npy")], "sq")
    report("SimLBR (retrained, 10k SD1.4)", [ld(f"runs/p4s/head_lbr_exact_seed{s}.pt") for s in (0, 1, 2)], sc("lbr_exact"), "sq")
    report("Supervised probe (SD1.4 images)", [ld(f"runs/p4s/head_sup_seed{s}.pt") for s in (0, 1, 2)], sc("sup"), "sq")
    report("Reconstruction probe (copies, final layer)", [ld(f"runs/p4s/head_recon_seed{s}.pt") for s in (0, 1, 2)], sc("recon"), "sq")
    S = dv.scores_for("genimage", rows, idx)
    report("Deja VAE-I (copies, layers 8-18)", [ld(f"runs/dv2/head_dv-single_seed{s}.pt") for s in (0, 1, 2)], [S["dv-single"][s][0] for s in (0, 1, 2)], "mid")
    dv.W, dv.FEAT = Path("runs/dv"), "cls24"
    S1 = dv.scores_for("genimage", rows, idx)
    report("Deja VAE-I (copies, final layer)", [ld(f"runs/dv/head_dv-single_seed{s}.pt") for s in (0, 1, 2)], [S1["dv-single"][s][0] for s in (0, 1, 2)], "fin")
    Path("runs/review_copies.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
