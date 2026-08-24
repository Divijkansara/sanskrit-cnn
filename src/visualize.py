"""
Step 4 of the pipeline: turn the numbers from train.py / evaluate.py into
figures for the README / report.

Produces, in outputs/figures/:
  * training_curves.png   - train/val loss and accuracy vs. epoch
  * confusion_matrix.png  - 46x46 normalised confusion matrix heatmap
  * sample_predictions.png- grid of test images with predicted vs true label
  * misclassified.png     - grid focused on the model's actual mistakes
  * dataset_sample.png    - one example per class, for context

Run:  python src/visualize.py   (after train.py and evaluate.py)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from dataset import get_class_names, load_splits


def _savefig(fig, name: str) -> None:
    path = cfg.FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {path}")


def plot_training_curves() -> None:
    rows = list(csv.DictReader(cfg.HISTORY_PATH.open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) * 100 for r in rows]
    va_acc = [float(r["val_acc"]) * 100 for r in rows]
    best_epoch = epochs[int(np.argmax(va_acc))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, tr_loss, marker="o", ms=3, label="train")
    ax1.plot(epochs, va_loss, marker="o", ms=3, label="validation")
    ax1.axvline(best_epoch, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, tr_acc, marker="o", ms=3, label="train")
    ax2.plot(epochs, va_acc, marker="o", ms=3, label="validation")
    ax2.axvline(best_epoch, color="gray", ls="--", lw=1, alpha=0.7,
                label=f"best epoch ({best_epoch})")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Training / validation performance", y=1.03, fontsize=13)
    _savefig(fig, "training_curves.png")


def plot_confusion_matrix() -> None:
    cm = np.load(cfg.METRIC_DIR / "confusion_matrix.npy")
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    names = get_class_names()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (row-normalised) — DHCD test set, 13,800 images")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("fraction of true class")
    _savefig(fig, "confusion_matrix.png")

    # A zoomed-in version showing only the classes the model confuses most,
    # which is far more legible than the full 46x46 grid.
    off_diag = cm_norm.copy()
    np.fill_diagonal(off_diag, 0)
    worst_true = np.argsort(off_diag.sum(axis=1))[::-1][:12]
    idx = sorted(set(worst_true.tolist()))
    sub = cm_norm[np.ix_(idx, idx)]
    sub_names = [names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(sub_names, rotation=90)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(sub_names)
    for i in range(len(idx)):
        for j in range(len(idx)):
            val = sub[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=7)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — 12 most-confused classes")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _savefig(fig, "confusion_matrix_top_confused.png")


def plot_sample_predictions(n: int = 24, misclassified_only: bool = False) -> None:
    splits = load_splits()
    x_test, y_test = splits["x_test"], splits["y_test"]
    names = get_class_names()

    preds_npz = np.load(cfg.METRIC_DIR / "test_predictions.npz")
    preds, probs = preds_npz["preds"], preds_npz["probs"]

    rng = np.random.default_rng(cfg.SEED)
    if misclassified_only:
        candidates = np.where(preds != y_test)[0]
        title = "Sample misclassifications — test set"
        fname = "misclassified.png"
    else:
        candidates = np.arange(len(y_test))
        title = "Sample predictions — test set"
        fname = "sample_predictions.png"

    if len(candidates) == 0:
        print(f"[viz] no examples for '{fname}', skipping")
        return

    chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)

    cols = 6
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, chosen):
        ax.imshow(x_test[idx], cmap="gray")
        ax.axis("off")
        true_name, pred_name = names[y_test[idx]], names[preds[idx]]
        conf = probs[idx, preds[idx]] * 100
        correct = y_test[idx] == preds[idx]
        color = "seagreen" if correct else "crimson"
        ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.0f}%)",
                     fontsize=8, color=color)
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02, fontsize=13)
    _savefig(fig, fname)


def plot_dataset_sample() -> None:
    splits = load_splits()
    x_train, y_train = splits["x_train"], splits["y_train"]
    names = get_class_names()

    fig, axes = plt.subplots(5, 10, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < cfg.NUM_CLASSES:
            idx = np.where(y_train == i)[0][0]
            ax.imshow(x_train[idx], cmap="gray")
            ax.set_title(names[i], fontsize=8)
    fig.suptitle("One example per class — DHCD training split", y=1.02, fontsize=13)
    _savefig(fig, "dataset_sample.png")


if __name__ == "__main__":
    plot_dataset_sample()
    plot_training_curves()
    plot_confusion_matrix()
    plot_sample_predictions(n=24, misclassified_only=False)
    plot_sample_predictions(n=24, misclassified_only=True)
