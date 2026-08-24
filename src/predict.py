"""
Classify unseen Sanskrit (Devanagari) character images with the trained model.

Accepts one or more image files (PNG/JPG/...), any size or color mode. Each
image is converted to grayscale, resized to 32x32, normalised the same way as
training, and passed through the network. Prints the top-3 predicted classes
with confidence for each image.

Usage:
    python src/predict.py samples/00_ka_0.png
    python src/predict.py samples/*.png
    python src/predict.py --topk 5 my_photo.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from model import build_model
from utils import get_device


def load_image_as_tensor(path: Path) -> torch.Tensor:
    """Read any image file and turn it into a normalised (1,1,32,32) tensor."""
    img = Image.open(path).convert("L")  # grayscale

    # DHCD characters are white strokes on a black background. A photographed
    # or scanned character is usually the opposite (dark ink on light paper),
    # so auto-detect and invert when the image is predominantly bright.
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - cfg.NORM_MEAN) / cfg.NORM_STD
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("class_names", cfg.CLASS_NAMES)


@torch.no_grad()
def predict_image(model, class_names, path: Path, device, topk: int = 3):
    tensor = load_image_as_tensor(path).to(device)
    probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    top_idx = probs.argsort()[::-1][:topk]
    return [(class_names[i], cfg.DEVANAGARI[i], float(probs[i])) for i in top_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="one or more image file paths")
    ap.add_argument("--checkpoint", default=str(cfg.BEST_MODEL_PATH), help="path to trained .pt checkpoint")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}\nRun `python src/train.py` first.")

    device = get_device()
    model, class_names = load_model(checkpoint, device)

    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"\n{path}: file not found, skipping")
            continue
        results = predict_image(model, class_names, path, device, topk=args.topk)
        print(f"\n{path.name}")
        for rank, (name, dev, prob) in enumerate(results, 1):
            marker = "->" if rank == 1 else "  "
            print(f"  {marker} #{rank}  {name:8s} ({dev})   {prob*100:5.1f}%")


if __name__ == "__main__":
    main()
