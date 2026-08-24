"""
Step 2 of the pipeline: train the CNN.

Trains SanskritCNN on the training split, evaluates on the validation split
after every epoch, keeps the checkpoint with the best validation accuracy, and
writes the full per-epoch history to outputs/metrics/training_history.csv.

The test split is never touched here -- it is only used by evaluate.py.

Run:  python src/train.py                 # defaults from config.py
      python src/train.py --epochs 30 --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from dataset import get_dataloaders
from model import build_model
from utils import AverageMeter, count_parameters, get_device, save_json, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    """One pass over `loader`. Training pass if `optimizer` is given."""
    training = optimizer is not None
    model.train(training)

    loss_meter, correct, total = AverageMeter(), 0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            loss_meter.update(loss.item(), labels.size(0))
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_meter.avg, correct / total


def train(args) -> dict:
    set_seed(cfg.SEED)
    device = get_device()
    torch.set_num_threads(args.threads)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers
    )

    model = build_model().to(device)
    n_params = count_parameters(model)

    print(f"[train] device={device} threads={args.threads}")
    print(f"[train] model parameters: {n_params:,}")
    print(f"[train] train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,}")
    print(f"[train] epochs={args.epochs} batch={args.batch_size} lr={args.lr}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY
    )
    # OneCycle: warm up then anneal. Converges in noticeably fewer epochs than a
    # flat learning rate, which matters when training on CPU.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
    )

    history = []
    best_val_acc, best_epoch, epochs_without_gain = 0.0, -1, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history.append(
            dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                 val_loss=va_loss, val_acc=va_acc, lr=lr_now, seconds=elapsed)
        )

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_without_gain = va_acc, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "num_classes": cfg.NUM_CLASSES,
                    "class_names": cfg.CLASS_NAMES,
                    "norm": {"mean": cfg.NORM_MEAN, "std": cfg.NORM_STD},
                },
                cfg.BEST_MODEL_PATH,
            )
            flag = "  <- best, saved"
        else:
            epochs_without_gain += 1

        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}%  |  "
            f"val loss {va_loss:.4f} acc {va_acc*100:5.2f}%  "
            f"({elapsed:.0f}s){flag}",
            flush=True,
        )

        # Persist history every epoch so a long run is never lost.
        write_history(history)

        if epochs_without_gain >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n[train] early stop: no val improvement for {cfg.EARLY_STOP_PATIENCE} epochs")
            break

    total_min = (time.time() - start) / 60
    summary = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "parameters": n_params,
        "batch_size": args.batch_size,
        "max_lr": args.lr,
        "weight_decay": cfg.WEIGHT_DECAY,
        "label_smoothing": cfg.LABEL_SMOOTHING,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "seed": cfg.SEED,
        "device": str(device),
        "total_minutes": round(total_min, 1),
        "checkpoint": str(cfg.BEST_MODEL_PATH.relative_to(cfg.PROJECT_ROOT)),
    }
    save_json(summary, cfg.METRIC_DIR / "train_summary.json")

    print(f"\n[train] done in {total_min:.1f} min")
    print(f"[train] best val accuracy {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"[train] checkpoint -> {cfg.BEST_MODEL_PATH}")
    return summary


def write_history(history: list) -> None:
    cfg.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cfg.HISTORY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Sanskrit character CNN.")
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
