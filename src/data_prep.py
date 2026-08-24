"""
Step 1 of the pipeline: acquire the dataset, preprocess it, and split it.

What this script does
---------------------
1. Downloads the DHCD (Devanagari Handwritten Character Dataset) from the
   authors' public GitHub repository, unless it is already present.
2. Loads the 92,000 grayscale character images and their integer labels.
3. Preprocesses every image: grayscale, resize to 32x32, scale to [0, 1].
4. Splits the official 78,200-image training pool into a stratified
   train / validation set; the official 13,800-image test set is kept
   completely untouched so it stays *unseen* by the model.
5. Computes the training-set pixel mean/std used for normalisation.
6. Writes `data/processed/dhcd_splits.npz` plus a `class_names.json`
   metadata file, and drops a handful of loose PNGs into `samples/`
   so `predict.py` can be demonstrated on real image files.

Run:  python src/data_prep.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from utils import save_json, set_seed


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> Path:
    """Fetch the raw DHCD .npz into data/raw/, returning its path."""
    if cfg.RAW_NPZ.exists() and not force:
        print(f"[data] raw dataset already present -> {cfg.RAW_NPZ}")
        return cfg.RAW_NPZ

    print(f"[data] cloning {cfg.DATASET_REPO} ...")
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "DHCD_Dataset"
        subprocess.run(
            ["git", "clone", "--depth", "1", cfg.DATASET_REPO, str(clone_dir)],
            check=True,
        )
        src = clone_dir / cfg.DATASET_NPZ_RELPATH
        if not src.exists():
            raise FileNotFoundError(f"expected {cfg.DATASET_NPZ_RELPATH} in the repo")
        cfg.RAW_NPZ.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, cfg.RAW_NPZ)

    size_mb = cfg.RAW_NPZ.stat().st_size / 1e6
    print(f"[data] saved raw dataset ({size_mb:.1f} MB) -> {cfg.RAW_NPZ}")
    return cfg.RAW_NPZ


# --------------------------------------------------------------------------- #
# 2. Preprocess
# --------------------------------------------------------------------------- #
def preprocess_images(images: np.ndarray, size: int = cfg.IMAGE_SIZE) -> np.ndarray:
    """
    Force every image to (size, size) uint8 grayscale.

    The DHCD images already arrive as 32x32 grayscale, so this is usually a
    no-op -- but it is written generically so the same function works if the
    dataset is swapped for one with mixed resolutions or RGB images.
    """
    if images.ndim == 4:  # (N, H, W, C) -> grayscale via luminance weights
        images = (
            0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
        ).astype(np.uint8)

    if images.shape[1] == size and images.shape[2] == size:
        return np.ascontiguousarray(images, dtype=np.uint8)

    print(f"[data] resizing {images.shape[1]}x{images.shape[2]} -> {size}x{size}")
    out = np.empty((images.shape[0], size, size), dtype=np.uint8)
    for i, img in enumerate(images):
        out[i] = np.asarray(
            Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8
        )
    return out


# --------------------------------------------------------------------------- #
# 3. Split + persist
# --------------------------------------------------------------------------- #
def build_splits(force_download: bool = False) -> dict:
    set_seed(cfg.SEED)
    npz_path = download_dataset(force=force_download)

    raw = np.load(npz_path)
    x_pool, y_pool = raw["arr_0"], raw["arr_1"]      # official train pool
    x_test, y_test = raw["arr_2"], raw["arr_3"]      # official held-out test

    # Raw labels are 1..46 -> convert to 0..45.
    y_pool = y_pool.astype(np.int64) - 1
    y_test = y_test.astype(np.int64) - 1

    x_pool = preprocess_images(x_pool)
    x_test = preprocess_images(x_test)

    print(f"[data] training pool : {x_pool.shape[0]:,} images")
    print(f"[data] test (unseen) : {x_test.shape[0]:,} images")
    print(f"[data] classes       : {len(np.unique(y_pool))}")

    # Stratified train/val split so every class keeps its proportion.
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool,
        y_pool,
        test_size=cfg.VAL_FRACTION,
        random_state=cfg.SEED,
        stratify=y_pool,
        shuffle=True,
    )

    # Normalisation statistics come from the TRAINING split only -- using the
    # val/test pixels here would leak information out of the held-out data.
    mean = float(x_train.astype(np.float32).mean() / 255.0)
    std = float(x_train.astype(np.float32).std() / 255.0)

    out_path = cfg.PROCESSED_DIR / "dhcd_splits.npz"
    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )

    meta = {
        "dataset": "DHCD - Devanagari Handwritten Character Dataset (Acharya et al., 2015)",
        "source": cfg.DATASET_REPO,
        "image_size": cfg.IMAGE_SIZE,
        "num_channels": cfg.NUM_CHANNELS,
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "devanagari": cfg.DEVANAGARI,
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "norm_mean": round(mean, 4),
        "norm_std": round(std, 4),
        "seed": cfg.SEED,
    }
    save_json(meta, cfg.PROCESSED_DIR / "class_names.json")

    print(f"[data] train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")
    print(f"[data] pixel mean={mean:.4f} std={std:.4f}")
    print(f"[data] wrote {out_path}")
    if abs(mean - cfg.NORM_MEAN) > 1e-3 or abs(std - cfg.NORM_STD) > 1e-3:
        print(
            f"[data] NOTE: update config.NORM_MEAN/NORM_STD to "
            f"{mean:.4f}/{std:.4f} to match this split."
        )

    export_sample_pngs(x_test, y_test, n_per_class=1)
    return meta


def export_sample_pngs(images: np.ndarray, labels: np.ndarray, n_per_class: int = 1) -> None:
    """Write a few real PNG files so predict.py can be run on image files."""
    written = 0
    for cls in range(cfg.NUM_CLASSES):
        idxs = np.where(labels == cls)[0][:n_per_class]
        for k, idx in enumerate(idxs):
            name = f"{cls:02d}_{cfg.CLASS_NAMES[cls]}_{k}.png"
            Image.fromarray(images[idx]).save(cfg.SAMPLE_DIR / name)
            written += 1
    print(f"[data] wrote {written} sample PNGs -> {cfg.SAMPLE_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and preprocess the DHCD dataset.")
    ap.add_argument("--force-download", action="store_true", help="re-download the raw archive")
    args = ap.parse_args()
    build_splits(force_download=args.force_download)
