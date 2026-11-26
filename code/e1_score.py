"""E1: score released detectors on the paper's fixed stratified subsets (the AEROBLADE rows), under each benchmark's original
and matched versions, so that every number is paired with the paper's own methods on the same rows.

Benchmark versions (row-aligned within each family; subset indices from runs/*/aeroblade/idx.npy):
  genimage_orig   GenImage files as distributed (JPEG reals, PNG generated)      genimage_q95  every image re-saved as JPEG q95
  cham            Chameleon as distributed          cham_c  processing-only control          cham_m  size-matched
  sb_q95          Synthbuster, downscaled reals, q95 re-save                    sb_crop_q95  cropped reals, q95 re-save
  sb_raw          Synthbuster, downscaled reals, files as stored (no re-save)
  eg_q95          EvalGEN (FLUX, GoT, Infinity, NOVA, OmniGen) + RAISE reals downscaled to 1024 (evalgen_prep.py), q95
  eg_crop_q95     the same generated images + RAISE reals cropped to 1024, q95
Each detector sees the full image after the version's re-save (if any) and applies its OWN preprocessing.
Third-party code lives in ~/anchor_data/ext/<name> (downloaded, untrusted; run with python -I). Scores are cached per
(image, re-save) so images shared between versions are scored once. Higher score = more likely generated.
    anchor/bin/python -I e1_score.py bfree [--sets ...]     -> runs/e1/<detector>/scores_<set>.npy (+ cache.json)
"""
import argparse, csv, io, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EXT = Path.home() / "anchor_data" / "ext"
SETS = {   # name: (manifest, subset index file, JPEG re-save quality or None)
    "genimage_orig": ("runs/p2/manifest.csv", "runs/p4s/aeroblade/idx.npy", None),
    "genimage_q95": ("runs/p2/manifest.csv", "runs/p4s/aeroblade/idx.npy", 95),
    "cham": ("runs/cham/manifest.csv", "runs/cham_off/aeroblade/idx.npy", None),
    "cham_c": ("runs/cham_c/manifest.csv", "runs/cham_off/aeroblade/idx.npy", None),
    "cham_m": ("runs/cham_m/manifest.csv", "runs/cham_off/aeroblade/idx.npy", None),
    "sb_q95": ("runs/sb/manifest.csv", "runs/sb/aeroblade/idx.npy", 95),
    "sb_crop_q95": ("runs/sb_crop/manifest.csv", "runs/sb/aeroblade/idx.npy", 95),
    "sb_raw": ("runs/sb/manifest.csv", "runs/sb/aeroblade/idx.npy", None),
    "eg_q95": ("runs/eg/manifest.csv", "runs/eg/idx_all.npy", 95),            # EvalGEN (2024-25 generators) + matched RAISE reals
    "eg_crop_q95": ("runs/eg_crop/manifest.csv", "runs/eg_crop/idx_all.npy", 95),
}


def rows_of(name):
    man, idx, q = SETS[name]
    R = list(csv.DictReader(open(ROOT / man))); I = np.load(ROOT / idx)
    P = [R[i]["path"] if os.path.isabs(R[i]["path"]) else str(ROOT / R[i]["path"]) for i in I]
    return I, P, q


def load(path, q):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if q:
        b = io.BytesIO(); im.save(b, format="JPEG", quality=int(q)); b.seek(0); im = Image.open(b).convert("RGB")
    return im


# ----------------------------------------------------------------------------- detectors (adapters)
def _use(repo_dir):
    """Put one repository first on sys.path and forget modules with generic names imported from another repository."""
    for k in [k for k in sys.modules if k.split(".")[0] in ("networks", "utils", "models", "data", "options")]:
        del sys.modules[k]
    sys.path.insert(0, str(repo_dir))


def make_bfree(dev):
    """B-Free (Guillaro et al., CVPR 2025), official code and weights (MD5-checked); licence accepted by the author of
    this study. Full image, ResNet normalisation, 5 crops of 504 px averaged in embedding space; output = logit."""
    import torch, yaml
    B = EXT / "bfree" / "code"; _use(B)
    from networks import get_network, load_weights
    from torchvision.transforms import Compose
    from utils.normalization import get_list_norm
    cfg = yaml.safe_load(open(B / "weights/BFREE_dino2reg4/config.yaml"))
    m = load_weights(get_network(cfg["arch"]), str(B / "weights/BFREE_dino2reg4" / cfg["weights_file"])).to(dev).eval()
    T = Compose(get_list_norm(cfg["norm_type"]))
    def score(im):
        with torch.no_grad():
            return float(m(T(im).unsqueeze(0).to(dev)).cpu().numpy()[:, 0].item())
    return score


def make_dda(dev):
    """DDA (Chen et al., NeurIPS 2025), official code and the checkpoint the repository names as the official release
    (Hugging Face Junwei-Xi/Dual-Data-Alignment, DDA_ckpt.pth). DINOv2 ViT-L/14 (torch.hub) with LoRA; centre crop 336
    (zero-padded if smaller, as torchvision does), CLIP normalisation; output = logit."""
    import torch
    from huggingface_hub import hf_hub_download
    from torchvision import transforms
    _use(EXT / "dda2" / "Inference")
    from models.dinov2_models_lora import DINOv2ModelWithLoRA
    m = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1, lora_targets=None)
    ck = torch.load(hf_hub_download("Junwei-Xi/Dual-Data-Alignment", "DDA_ckpt.pth"), map_location="cpu")
    m.load_state_dict(ck["model"]); m = m.to(dev).eval()
    T = transforms.Compose([transforms.CenterCrop(336), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    def score(im):
        with torch.no_grad():
            return float(m(T(im).unsqueeze(0).to(dev)).flatten()[0].item())
    return score


def make_npr(dev):
    """NPR (Tan et al., CVPR 2024), official code and NPR.pth (in the repository; ProGAN 4-class training). As in the
    authors' test.py for datasets with mixed sizes: resize to 256 x 256, no crop, ImageNet normalisation; output = logit."""
    import torch
    from torchvision import transforms
    _use(EXT / "npr")
    from networks.resnet import resnet50
    sd = torch.load(EXT / "npr" / "NPR.pth", map_location="cpu")["model"]
    m = resnet50(num_classes=1); m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=True)
    m = m.to(dev).eval()
    T = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def score(im):
        with torch.no_grad():
            return float(m(T(im).unsqueeze(0).to(dev)).flatten()[0].item())
    return score


def make_alignedforensics(dev):
    """AlignedForensics (Rajan et al., ICLR 2025), official test code and the "Ours" checkpoint (Google Drive link in the
    README). Config as in the repository's example: arch res50nodown, ResNet normalisation, patch_size null (full image,
    no resize); output = logit. DEVIATION (memory): images with a side above 1,024 px are centre-cropped to at most
    1,024 x 1,024 at native resolution (no resampling); full-resolution inference on 2688 x 1536 images swaps on the 16 GB
    laptop (about 100 s per image). The network ends in global average pooling, so the crop keeps its input scale."""
    import torch
    _use(EXT / "alignedforensics" / "test_code")
    from networks import create_architecture, load_weights
    from utils.processing import make_normalize
    m = load_weights(create_architecture("res50nodown"), str(EXT / "alignedforensics" / "weights" / "ours" / "ours.pth")).eval()
    m = m.to(dev)
    T = make_normalize("resnet")
    def score(im):
        w, h = im.size
        if max(w, h) > 1024:
            cw, ch = min(w, 1024), min(h, 1024); l, t = (w - cw) // 2, (h - ch) // 2
            im = im.crop((l, t, l + cw, t + ch))
        with torch.no_grad():
            y = float(m(T(im).unsqueeze(0).to(dev)).flatten()[0].item())
        if dev == "mps":                          # images vary in size: release cached GPU blocks, or memory grows into swap
            torch.mps.empty_cache()
        return y
    return score


def make_spai(dev):
    """SPAI (Karageorgiou et al., CVPR 2025), official code and checkpoint (Google Drive link in the README), configs/spai.yaml.
    As in its `infer` command: original resolution (no resize), reflect-padded to at least 224 px (albumentations
    PadIfNeeded default), scaled to [0, 1], arbitrary-resolution forward with FEATURE_EXTRACTION_BATCH patches; output = logit.
    Our wrapper replaces the CSV data loader and the hard-coded .cuda() calls; it checks that every model weight is loaded."""
    import logging, tempfile, torch
    _use(EXT / "spai")
    for k in [k for k in sys.modules if k == "spai" or k.startswith("spai.")]:
        del sys.modules[k]
    from spai.config import get_config
    from spai.models.build import build_cls_model
    from spai.utils import load_pretrained
    ck = EXT / "spai" / "weights" / "spai.pth"
    cfg = get_config({"cfg": str(EXT / "spai" / "configs" / "spai.yaml"), "batch_size": 1, "test_csv": [], "test_csv_root": [],
                      "lmdb_path": None, "output": tempfile.mkdtemp(), "tag": "spai", "pretrained": str(ck), "resize_to": None, "opts": ()})
    m = build_cls_model(cfg)
    load_pretrained(cfg, m, logging.getLogger("spai"), checkpoint_path=ck, verbose=False)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    missing = [k for k in m.state_dict() if k not in sd]
    assert not missing, f"SPAI weights not loaded: {missing[:5]}"
    # Patches (224 px, stride 224) pass through the ViT independently; the config's batch of 400 patches only sets how many
    # go through at once. 400 exhausts the 16 GB laptop's GPU memory on large images, so we use 64 (scores unchanged; checked).
    m = m.to(dev).eval(); nb = min(cfg.MODEL.FEATURE_EXTRACTION_BATCH, 64)
    def score(im):
        x = np.asarray(im, dtype=np.float32) / 255.0
        h, w = x.shape[:2]
        if h < 224 or w < 224:                    # albumentations PadIfNeeded: centred, BORDER_REFLECT_101
            ph, pw = max(224 - h, 0), max(224 - w, 0)
            x = np.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), mode="reflect")
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0).to(dev)   # [1, C, H, W], as after the loader's views squeeze
        with torch.no_grad():
            y = float(m([t], nb)[0, 0].item())
        if dev == "mps":
            torch.mps.empty_cache()
        return y
    return score


DETECTORS = {"bfree": make_bfree, "dda": make_dda, "npr": make_npr, "alignedforensics": make_alignedforensics, "spai": make_spai}


def main():
    import torch
    a = argparse.ArgumentParser(); a.add_argument("detector", choices=list(DETECTORS)); a.add_argument("--sets", nargs="+", default=list(SETS))
    a = a.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = ROOT / "runs" / "e1" / a.detector; out.mkdir(parents=True, exist_ok=True)
    cf = out / "cache.json"; cache = json.load(open(cf)) if cf.exists() else {}
    score = DETECTORS[a.detector](dev)
    for name in a.sets:
        f = out / f"scores_{name}.npy"
        if f.exists():
            continue
        I, P, q = rows_of(name); S = np.zeros(len(P), np.float32); t0 = time.time(); new = 0
        for k, p in enumerate(P):
            key = f"{p}|q{q}"
            if key not in cache:
                cache[key] = score(load(p, q)); new += 1
                if new % 200 == 0:
                    json.dump(cache, open(cf, "w"))
                    print(f"  {a.detector} {name} {k + 1}/{len(P)}  {(time.time() - t0) / new:.2f} s/image", flush=True)
            S[k] = cache[key]
        json.dump(cache, open(cf, "w")); np.save(f, S); np.save(out / f"idx_{name}.npy", I)
        print(f"{a.detector} {name} done ({new} new images)", flush=True)
    print(f"E1 {a.detector} FINISHED", flush=True)


if __name__ == "__main__":
    main()
