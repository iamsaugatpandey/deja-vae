"""E1c: does the mid-layer peak of the fixed-point signal depend on SELF-SUPERVISED training?

DINOv3 and DINOv2 are both self-supervised. CLIP ViT-L/14 (openai/clip-vit-large-patch14; image-text contrastive,
24 layers like the DINO ViT-Ls, 224 px, CLIP normalisation, class token = token 0) is measured on the SAME images
(runs/dv/layers_full/groups.json) with the SAME ft-MSE copies (deterministic: latent mode), raw condition only,
plus the simulated-fake and blur-control rows for the reals. Output: runs/dv/layers_full/clip/*.npz + report.txt
(AUROC per layer; same conventions as dv_layers_full.py: stored values are distances, AUROC negates).
    anchor/bin/python dv_layers_clip.py run | report
"""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dv_layers_full as LF

OUT = LF.OUT / "clip"; OUT.mkdir(parents=True, exist_ok=True)
CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class ClipProbe(LF.Probe):
    def __init__(self):
        import torch
        from transformers import CLIPVisionModel
        self.t = torch
        self.enc = ap.Encoders(ap.pick_device(), need_vae=True, norm="simlbr")   # ft-MSE autoencoder (DINOv3 unused)
        dev = self.enc.device
        m = CLIPVisionModel.from_pretrained(CLIP_ID).to(dev).eval()
        self.models = {"clip": (m, torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1),
                                torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1), 224, 1)}


def cmd_run():
    P = ClipProbe(); G = json.load(open(LF.OUT / "groups.json"))
    for name, paths in G.items():
        f = OUT / f"{name.replace(':', '_')}_raw.npz"
        if f.exists():
            continue
        t0 = time.time(); ims = [LF.make_view(p, "raw") for p in paths]; R = P.recon(ims)
        out = {f"x/{k}": v for k, v in P.measure(ims, R).items()}
        if name == "gi:real":
            out.update({f"sim/{k}": v for k, v in P.measure(R, P.recon(R)).items()})
            D = [ap.blur_to_psnr(a, ap.psnr(a, b)) for a, b in zip(ims, R)]
            out.update({f"blur/{k}": v for k, v in P.measure(D, P.recon(D)).items()})
        np.savez(f, **out); print(f"clip {name:32s} n={len(ims)} {time.time() - t0:5.0f}s", flush=True)
    print("CLIP LAYERS RUN DONE", flush=True)
    cmd_report()


def cmd_report():
    import pandas as pd
    L = lambda n: dict(np.load(OUT / f"{n}_raw.npz"))
    real = L("gi_real"); gens = sorted(f.stem[3:-4] for f in OUT.glob("gi_*_raw.npz") if f.stem != "gi_real_raw")
    tab = {}
    for k in [k[2:] for k in real if k.startswith("x/")]:
        row = {g: LF.auroc(L(f"gi_{g}")[f"x/{k}"], real[f"x/{k}"]) for g in gens}
        row["MEAN"] = np.mean(list(row.values()))
        row["simR(x)"] = LF.auroc(real[f"sim/{k}"], real[f"x/{k}"]); row["blur*"] = LF.auroc(real[f"blur/{k}"], real[f"x/{k}"])
        row["CHAM"] = LF.auroc(L("cham_fake")[f"x/{k}"], L("cham_real")[f"x/{k}"])
        tab[k] = row
    df = pd.DataFrame(tab).T.round(3)
    cls = df[[i.startswith("clip/cls") for i in df.index]]
    txt = "\n".join(["E1c CLIP ViT-L/14 layer diagnostic (raw copies; same images as dv_layers_full).", df.sort_index().to_string(), "",
                     f"best CLS layer {cls['MEAN'].idxmax()} MEAN {cls['MEAN'].max():.3f}; final clip/cls@24 MEAN {df.loc['clip/cls@24', 'MEAN']:.3f}; "
                     f"CHAM best {cls['CHAM'].idxmax()} {cls['CHAM'].max():.3f}"])
    print(txt); (OUT / "report.txt").write_text(txt + "\n")


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
