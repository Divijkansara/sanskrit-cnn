"""
Step 3 of the pipeline: evaluate the trained model on the unseen test split.

Loads the best checkpoint from train.py and the official DHCD test set
(13,800 images that never appear in training or validation), then reports:
  * overall accuracy
  * macro- and weighted-average precision / recall / F1
  * a full per-class classification report
  * a confusion matrix (raw counts + a normalised version)

All numeric results are written to outputs/metrics/ so they can be embedded
in the README and cited in the report. Plot generation lives in visualize.py.

Run:  python src/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from dataset import get_dataloaders, get_class_names
from model import build_model
from utils import get_device, save_json


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def evaluate(checkpoint_path: Path = cfg.BEST_MODEL_PATH) -> dict:
    device = get_device()
    class_names = get_class_names()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val acc {ckpt['val_acc']*100:.2f}%)")

    _, _, test_loader = get_dataloaders(num_workers=0)
    print(f"[eval] scoring {len(test_loader.dataset):,} unseen test images ...")

    preds, labels, probs = collect_predictions(model, test_loader, device)
    np.savez_compressed(
        cfg.METRIC_DIR / "test_predictions.npz",
        preds=preds, labels=labels, probs=probs,
    )

    accuracy = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=list(range(cfg.NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    np.save(cfg.METRIC_DIR / "confusion_matrix.npy", cm)
    (cfg.METRIC_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_json(report_dict, cfg.METRIC_DIR / "classification_report.json")

    summary = {
        "checkpoint_epoch": ckpt["epoch"],
        "test_images": int(len(labels)),
        "accuracy": accuracy,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
    }
    save_json(summary, cfg.METRIC_DIR / "test_summary.json")

    # Which classes are hardest? Useful for the README + error analysis.
    per_class_f1 = [
        (class_names[i], report_dict[class_names[i]]["f1-score"], report_dict[class_names[i]]["support"])
        for i in range(cfg.NUM_CLASSES)
    ]
    per_class_f1.sort(key=lambda t: t[1])
    worst = per_class_f1[:5]

    print(f"\n[eval] accuracy            {accuracy*100:.2f}%")
    print(f"[eval] precision (macro)   {prec_macro*100:.2f}%")
    print(f"[eval] recall    (macro)   {rec_macro*100:.2f}%")
    print(f"[eval] f1        (macro)   {f1_macro*100:.2f}%")
    print(f"[eval] precision (weighted) {prec_weighted*100:.2f}%")
    print(f"[eval] recall    (weighted) {rec_weighted*100:.2f}%")
    print(f"[eval] f1        (weighted) {f1_weighted*100:.2f}%")
    print("\n[eval] 5 lowest-F1 classes:")
    for name, f1, support in worst:
        print(f"   {name:8s} f1={f1:.3f}  support={int(support)}")
    print(f"\n[eval] wrote metrics -> {cfg.METRIC_DIR}")

    return summary


if __name__ == "__main__":
    evaluate()
