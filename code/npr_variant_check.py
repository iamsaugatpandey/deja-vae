"""Reproduction check for NPR: the authors' test.py has two preprocessing settings (resize to 256 x 256 for datasets of
mixed sizes, which e1_score.py uses; or no resize with a 224 centre crop). Scores EvalGEN's generated images (as
distributed) and the Chameleon E1 subset under the second setting, for comparison with third-party published numbers.
    anchor/bin/python -I npr_variant_check.py   -> runs/e1/npr/scores_<set>_crop224.npy
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_score as E
import torch
from torchvision import transforms


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    E._use(E.EXT / "npr")
    from networks.resnet import resnet50
    sd = torch.load(E.EXT / "npr" / "NPR.pth", map_location="cpu")["model"]
    m = resnet50(num_classes=1); m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=True)
    m = m.to(dev).eval()
    T = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for name in ("eg_raw_fake", "cham", "sb_raw"):
        I, P, q = E.rows_of(name); out = []
        for p in P:
            with torch.no_grad():
                out.append(float(m(T(E.load(p, q)).unsqueeze(0).to(dev)).flatten()[0].item()))
        np.save(E.ROOT / f"runs/e1/npr/scores_{name}_crop224.npy", np.array(out)); print(name, len(out), flush=True)


if __name__ == "__main__":
    main()
