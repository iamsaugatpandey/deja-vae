"""Review: test-time cost per image on the 16 GB laptop (Apple M-series, PyTorch MPS), 64 GenImage images, batch 16,
after one warm-up batch. Each method's own test-time path (no feature caching):
  SimLBR (released) / supervised / Deja VAE-I: one DINOv3 ViT-L/16 forward at 256 px (Deja VAE-I reads layers 8-18).
  Deja VAE-P / -TF: ft-MSE encode+decode of the native 256 crop, then DINOv3 on image and copy.
  RIGID: DINOv2 ViT-L/14 forward on image and noisy image.
  AEROBLADE: encode+decode with 3 autoencoders (SD1, SD2, Kandinsky 2.1 MoVQ) + LPIPS (VGG) for each.
    anchor/bin/python latency.py   -> runs/latency.txt
    anchor/bin/python latency.py aero 4   -> re-times AEROBLADE alone at batch 4 (appended; batch 16 swaps on 16 GB)
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import anchor_pilot as ap
import dejavae as dv


def main():
    import torch
    dev = ap.pick_device()
    rows = [r for r in ap.read_manifest(Path("runs/p2/manifest.csv")) if r["split"] == "test"][::150]
    rows = [r for r in rows if dv.view(r["path"]).size == (256, 256)][:64]
    views = [dv.view(r["path"]) for r in rows]; sq = [ap.load_rgb(r["path"], 256, "simlbr") for r in rows]
    sync = (lambda: torch.mps.synchronize()) if dev.type == "mps" else (lambda: None)

    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    def timeit(fn, ims):
        fn(ims[:bs]); sync(); t = time.time()
        for s in range(0, len(ims), bs):
            fn(ims[s:s + bs])
        sync(); ms = (time.time() - t) / len(ims) * 1000; print(f"  timed {ms:.1f} ms/image", flush=True); return ms
    enc = ap.Encoders(dev, norm="simlbr", vae=dv.AE); P = dv.Pairs(enc)
    out = {}
    only_aero = len(sys.argv) > 1 and sys.argv[1] == "aero"
    if not only_aero:
      out["SimLBR / supervised probe (DINOv3, final layer)"] = timeit(lambda x: enc.cls(x), sq)
      out["Deja VAE-I (DINOv3, layers 8-18)"] = timeit(lambda x: P.phi(x), views)
      out["Deja VAE-P / -TF (ft-MSE copy + 2 DINOv3 passes)"] = timeit(lambda x: (P.phi(x), P.phi(P.recon(x))), views)
    if not only_aero:
      from transformers import AutoModel
      d2 = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
      def rigid(x):
          with torch.no_grad():
              t = enc.to_tensor([im.resize((224, 224)) for im in x])
              d2(pixel_values=t); d2(pixel_values=(t + 0.05 * torch.randn_like(t)))
      out["RIGID (DINOv2, 2 passes)"] = timeit(rigid, views)
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    aes = [ap.load_autoencoder(s, dev) for s in ("stable-diffusion-v1-5/stable-diffusion-v1-5:vae",
                                                    "sd-research/stable-diffusion-2-1-base:vae",
                                                    "kandinsky-community/kandinsky-2-1:movq")]
    def aero(x):
        with torch.no_grad():
            t = enc.to_tensor(x) * 2 - 1
            for ae in aes:
                e = ae.encode(t); z = e.latent_dist.mode() if hasattr(e, "latent_dist") else (e.latents if hasattr(e, "latents") else e.latent)
                r = ae.decode(z).sample; lp(t, r.clamp(-1, 1))
    out["AEROBLADE (3 autoencoders + LPIPS)"] = timeit(aero, views)
    if only_aero:   # batch 16 with three autoencoders + VGG exceeds the 16 GB laptop's memory and swaps
        txt = f"  AEROBLADE re-timed at batch {bs}                     {out['AEROBLADE (3 autoencoders + LPIPS)']:7.1f}"
        print(txt); open("runs/latency.txt", "a").write(txt + "\n"); return
    txt = "\n".join([f"Test-time cost per image (ms), {dev.type}, batch {bs}, 256-px views"] + [f"  {k:52s} {v:7.1f}" for k, v in out.items()])
    print(txt); Path("runs/latency.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
