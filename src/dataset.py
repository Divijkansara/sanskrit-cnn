"""
Torch Dataset + DataLoader construction for the preprocessed DHCD splits.

Augmentation (small rotations / translations / scaling) is applied to the
training split only. Devanagari characters are not rotation-invariant, so the
transforms are deliberately mild -- enough to model natural handwriting
variation without turning one character into another.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from utils import load_json


SPLITS_PATH = cfg.PROCESSED_DIR / "dhcd_splits.npz"


class SanskritCharDataset(Dataset):
    """Wraps one preprocessed split (images uint8 HxW, labels int64 0..45)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]                       # (H, W) uint8
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W) uint8
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(self.labels[idx])


def build_transforms(train: bool) -> transforms.Compose:
    """uint8 tensor -> float, augment (train only), normalise."""
    steps = []
    if train:
        steps.append(
            transforms.RandomAffine(
                degrees=8,                    # slight slant
                translate=(0.08, 0.08),       # off-centre writing
                scale=(0.92, 1.08),           # thicker / thinner strokes
                shear=5,
                fill=0,                       # background is black
            )
        )
    steps += [
        transforms.ConvertImageDtype(torch.float32),      # uint8 -> [0, 1]
        transforms.Normalize((cfg.NORM_MEAN,), (cfg.NORM_STD,)),
    ]
    return transforms.Compose(steps)


def load_splits(path: Path = SPLITS_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/data_prep.py` first."
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    d = load_splits()
    train_ds = SanskritCharDataset(d["x_train"], d["y_train"], build_transforms(True))
    val_ds = SanskritCharDataset(d["x_val"], d["y_val"], build_transforms(False))
    test_ds = SanskritCharDataset(d["x_test"], d["y_test"], build_transforms(False))
    return train_ds, val_ds, test_ds


def get_dataloaders(
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = get_datasets()
    common = dict(num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def get_class_names() -> list:
    meta_path = cfg.PROCESSED_DIR / "class_names.json"
    if meta_path.exists():
        return load_json(meta_path)["class_names"]
    return cfg.CLASS_NAMES


if __name__ == "__main__":
    tr, va, te = get_dataloaders(num_workers=0)
    xb, yb = next(iter(tr))
    print("batch:", xb.shape, xb.dtype, "labels:", yb.shape, yb[:8].tolist())
    print(f"train {len(tr.dataset):,} | val {len(va.dataset):,} | test {len(te.dataset):,}")
    print("normalised range:", float(xb.min()), "->", float(xb.max()))
