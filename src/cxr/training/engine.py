"""Training / evaluation loops with AMP and cosine decay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from cxr.models.densenet import DenseNet121Classifier
from cxr.models.losses import build_loss, pos_weight_from_labels
from cxr.training.metrics import compute_metrics, format_metrics_table, save_metrics


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        ys.append(batch["target"].numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return compute_metrics(y_true, y_score, threshold=threshold)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, accum_steps: int, clip_grad: float) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets) / accum_steps
        scaler.scale(loss).backward()
        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * accum_steps
    return running / max(1, len(loader))


def fit(cfg, loaders: dict) -> DenseNet121Classifier:
    device = _device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = DenseNet121Classifier(
        pretrained=cfg.get("model.pretrained", "imagenet"),
        dropout=float(cfg.get("model.dropout", 0.2)),
    ).to(device)

    freeze_epochs = int(cfg.get("train.freeze_backbone_epochs", 1))
    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"Frozen backbone for first {freeze_epochs} epoch(s)")

    pos_weight = None
    if cfg.get("train.use_pos_weight", True) and "train" in loaders:
        labels = loaders["train"].dataset.labels
        pos_weight = pos_weight_from_labels(labels).to(device)
        print("pos_weight:", {i: f"{w:.2f}" for i, w in enumerate(pos_weight.tolist())})

    criterion = build_loss(pos_weight=pos_weight)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.get("train.lr", 1e-4)),
        weight_decay=float(cfg.get("train.weight_decay", 1e-5)),
    )
    epochs = int(cfg.get("train.epochs", 12))
    warmup = int(cfg.get("train.warmup_epochs", 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=float(cfg.get("train.min_lr", 1e-6)))
    if warmup > 0:
        warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        scheduler = SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])
    else:
        scheduler = cosine

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("train.amp", True)) and device.type == "cuda")
    ckpt_dir = cfg.path("checkpoint_dir")
    out_dir = cfg.path("output_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    patience = int(cfg.get("train.early_stop_patience", 4))
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = AdamW(
                model.parameters(),
                lr=float(cfg.get("train.lr", 1e-4)),
                weight_decay=float(cfg.get("train.weight_decay", 1e-5)),
            )
            remaining = epochs - freeze_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=float(cfg.get("train.min_lr", 1e-6)))
            print("Unfroze backbone")

        print(f"\n=== epoch {epoch}/{epochs}  lr={optimizer.param_groups[0]['lr']:.2e} ===")
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            criterion,
            device,
            accum_steps=int(cfg.get("train.accum_steps", 2)),
            clip_grad=float(cfg.get("train.clip_grad", 1.0)),
        )
        val_metrics = evaluate(
            model,
            loaders.get("val") or loaders["train"],
            device,
            threshold=float(cfg.get("eval.threshold", 0.5)),
        )
        scheduler.step()
        auc = val_metrics["mean_roc_auc"]
        print(f"train_loss={train_loss:.4f}  val_mean_auc={auc:.4f}")
        print(format_metrics_table(val_metrics))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_auc": auc})

        extra = {"epoch": epoch, "val_mean_auc": auc, "history": history}
        model.save(ckpt_dir / "last.pt", extra=extra)
        if auc > best_auc:
            best_auc = auc
            bad = 0
            model.save(ckpt_dir / "best.pt", extra=extra)
            save_metrics(val_metrics, out_dir / "val_metrics.json")
            print(f"Saved new best checkpoint  AUC={best_auc:.4f}")
        else:
            bad += 1
            print(f"No improvement ({bad}/{patience})")
            if bad >= patience:
                print("Early stopping")
                break

    print(f"\nBest val mean ROC-AUC: {best_auc:.4f}")
    return model
