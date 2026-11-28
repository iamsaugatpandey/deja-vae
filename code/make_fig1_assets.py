"""Assets for the overview figure, panel a, from images that may be published under an open licence.
Real: "Wimbachgries in the Wimbach Valley" (Wikimedia Commons, user Mataksim, own work, CC0; Sony ILCE-6400A), brought
to 1024 x 1024 as in the downscaled matching (centre crop to the aspect ratio, bicubic resize).
Generated: SDXL base 1.0, 1024 x 1024, decoded with the checkpoint's 1.0 autoencoder (`vae_1_0`), from a description of
that photograph, as Synthbuster prompts its generators from descriptions of RAISE photographs. Three seeds; the image
whose own-decoder distance is the median of the three is used, so that the panel is not a best case.
Measurement as in dv_bank_sdxl.py: the native-resolution 256 x 256 centre view, an autoencoder round trip with
SDXL 1.0 (`vae_1_0`) and SDXL 0.9 (`vae`), and 1 - cos of the DINOv3 class token at every layer.
    anchor/bin/python -I make_fig1_assets.py <real photo>   -> paper/figures/fig1_assets_open/
"""
import os, sys
os.environ.setdefault("DV_FEAT", "mid")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchor_pilot as ap
import dv_bank_sb as B
from dejavae import to256

OUT = Path(__file__).resolve().parent / "figures/fig1_assets_open"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = ("a photograph of a mountain valley in clear daylight: steep slopes covered in dense pine forest, grey limestone "
          "cliffs, a wide pale gravel riverbed on the valley floor, distant blue mountain ridges")
SEEDS = (1, 2, 3)


def square1024(im):
    from PIL import Image
    w, h = im.size; s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((1024, 1024), Image.BICUBIC)


def view256(im):
    w, h = im.size; l, t = (w - 256) // 2 // 8 * 8, (h - 256) // 2 // 8 * 8
    return im.crop((l, t, l + 256, t + 256))


def generate(dev):
    import torch
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.vae = AutoencoderKL.from_pretrained(SDXL, subfolder="vae_1_0", torch_dtype=torch.float32)
    pipe.upcast_vae = lambda: None                   # the VAE is already float32
    pipe = pipe.to(dev); ims = []
    for s in SEEDS:
        with torch.no_grad():
            lat = pipe(PROMPT, num_inference_steps=30, guidance_scale=6.0, height=1024, width=1024,
                       generator=torch.Generator("cpu").manual_seed(s), output_type="latent").images
            lat = lat.to(torch.float32) / pipe.vae.config.scaling_factor
            x = pipe.vae.decode(lat).sample
        x = ((x.float().clamp(-1, 1) + 1) / 2 * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        ims.append(Image.fromarray(x)); ims[-1].save(OUT / f"generated_seed{s}.png"); print("generated seed", s, flush=True)
    del pipe
    if dev.type == "mps":
        torch.mps.empty_cache()
    return ims


def main():
    import torch
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True); dev = ap.pick_device()
    real = square1024(Image.open(sys.argv[1]).convert("RGB")); real.save(OUT / "real_1024.png")
    gens = [Image.open(OUT / f"generated_seed{s}.png").convert("RGB") for s in SEEDS] \
        if all((OUT / f"generated_seed{s}.png").exists() for s in SEEDS) else generate(dev)
    enc = ap.Encoders(dev, need_vae=False, norm="simlbr")

    def cls_all(ims):
        with torch.no_grad():
            x = (enc.to_tensor([to256(im) for im in ims]) - enc.mean) / enc.std
            hs = enc.dino(pixel_values=x, output_hidden_states=True).hidden_states
        return torch.stack([h[:, 0] for h in hs], 1).float().cpu().numpy()      # [n, 25, dim]

    views = [view256(real)] + [view256(g) for g in gens]
    copies = {}
    for tag, sub in (("v10", "vae_1_0"), ("v09", "vae")):
        enc.vae = ap.load_autoencoder(f"{SDXL}:{sub}", dev); copies[tag] = B.reconstruct(enc, views); enc.vae = None
    P = cls_all(views); D = {}
    for tag in copies:
        Q = cls_all(copies[tag])
        cos = (P * Q).sum(-1) / (np.linalg.norm(P, axis=-1) * np.linalg.norm(Q, axis=-1))
        D[tag] = 1 - cos                                                                   # [n, 25]
    mid = list(range(8, 19, 2))
    own = D["v10"][1:, mid].mean(1); k = int(np.argsort(own)[len(own) // 2])               # median seed
    print("own-decoder distance per seed", np.round(own, 4).tolist(), "-> seed", SEEDS[k])
    i = 1 + k
    views[0].save(OUT / "real_view.png"); views[i].save(OUT / "fake_view.png")
    real.save(OUT / "real_full.png"); gens[k].save(OUT / "fake_full.png")              # shown with the crop outlined
    for tag in copies:
        copies[tag][0].save(OUT / f"real_copy_{tag}.png"); copies[tag][i].save(OUT / f"fake_copy_{tag}.png")
    np.savez(OUT / "layer_dist.npz", real_v10=D["v10"][0], real_v09=D["v09"][0], fake_v10=D["v10"][i], fake_v09=D["v09"][i])
    m = {n: float(np.mean(a[mid])) for n, a in (("real_v10", D["v10"][0]), ("fake_v10", D["v10"][i]), ("fake_v09", D["v09"][i]))}
    print({k: round(v, 4) for k, v in m.items()}, "ratio 0.9/1.0:", round(m["fake_v09"] / m["fake_v10"], 2))
    (OUT / "SOURCES.txt").write_text(
        "real_*: 'Wimbachgries in the Wimbach Valley', Wikimedia Commons, by Mataksim (own work), CC0 1.0 public-domain "
        "dedication; https://commons.wikimedia.org/wiki/File:Wimbachgries_in_the_Wimbach_Valley.jpg\n"
        f"generated_* / fake_*: generated by the author with SDXL base 1.0 (CreativeML Open RAIL++-M; the licence claims "
        f"no rights in outputs), seeds {SEEDS}, prompt: {PROMPT!r}; decoded with the vae_1_0 autoencoder.\n")


if __name__ == "__main__":
    main()
