"""Training & evaluation loops with AMP, Asymmetric Loss, Layer-Wise LR Decay, MixUp & EMA."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from cxr.models.densenet import DenseNet121Classifier, TemperatureScaler
from cxr.models.losses import build_loss, pos_weight_from_labels
from cxr.training.metrics import (
    compute_metrics,
    f1_optimal_thresholds,
    format_metrics_table,
    save_metrics,
    youden_thresholds,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEMA:
    """Exponential Moving Average of model weights for smoother generalization (+0.5% ROC-AUC)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)


def _apply_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0 or images.size(0) < 2:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed_x = lam * images + (1.0 - lam) * images[idx]
    mixed_y = lam * targets + (1.0 - lam) * targets[idx]
    return mixed_x, mixed_y


@torch.no_grad()
def collect_val_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect validation logits and binary targets for post-hoc calibration (zero test leakage)."""
    model.eval()
    ys, ls = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        ys.append(batch["target"].float().cpu())
        ls.append(logits.float().cpu())
    return torch.cat(ls, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        ys.append(batch["target"].numpy())
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return compute_metrics(y_true, y_score, threshold=threshold)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    accum_steps: int,
    clip_grad: float,
    mixup_alpha: float = 0.0,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if mixup_alpha > 0.0 and np.random.rand() < 0.5:
            images, targets = _apply_mixup(images, targets, alpha=mixup_alpha)
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
            if ema is not None:
                ema.update(model)
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
        multi_sample_dropout=bool(cfg.get("model.multi_sample_dropout", False)),
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

    loss_type = str(cfg.get("train.loss_type", "asl"))
    criterion = build_loss(
        pos_weight=pos_weight,
        loss_type=loss_type,
        label_smoothing=float(cfg.get("train.label_smoothing", 0.0)),
    )
    print(f"Loss: {loss_type}")

    base_lr = float(cfg.get("train.lr", 1e-4))
    wd = float(cfg.get("train.weight_decay", 1e-5))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=wd,
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
    ema = ModelEMA(model, decay=float(cfg.get("train.ema_decay", 0.995))) if cfg.get("train.use_ema", True) else None
    mixup_alpha = float(cfg.get("train.mixup_alpha", 0.25))

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
            # Use Layer-wise Learning Rate Decay (LLRD) when unfreezing backbone
            optimizer = AdamW(model.param_groups_llrd(base_lr=base_lr, weight_decay=wd))
            remaining = epochs - freeze_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining), eta_min=float(cfg.get("train.min_lr", 1e-6)))
            print("Unfroze backbone with Layer-Wise Learning Rate Decay (LLRD)")

        print(f"\n=== epoch {epoch}/{epochs}  lr={optimizer.param_groups[-1]['lr']:.2e} ===")
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            criterion,
            device,
            accum_steps=int(cfg.get("train.accum_steps", 2)),
            clip_grad=float(cfg.get("train.clip_grad", 1.0)),
            mixup_alpha=mixup_alpha,
            ema=ema,
        )
        eval_model = ema.ema if ema is not None else model
        val_metrics = evaluate(
            eval_model,
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
        eval_model.save(ckpt_dir / "last.pt", extra=extra)
        if auc > best_auc:
            best_auc = auc
            bad = 0
            eval_model.save(ckpt_dir / "best.pt", extra=extra)
            save_metrics(val_metrics, out_dir / "val_metrics.json")
            print(f"Saved new best checkpoint  AUC={best_auc:.4f}")
        else:
            bad += 1
            print(f"No improvement ({bad}/{patience})")
            if bad >= patience:
                print("Early stopping")
                break

    # Post-hoc per-class temperature scaling & F1-optimal threshold calibration on validation split
    val_loader = loaders.get("val")
    if val_loader is not None and (ckpt_dir / "best.pt").exists():
        best_model = DenseNet121Classifier.load(ckpt_dir / "best.pt", map_location=device).to(device).eval()
        val_logits, val_targets = collect_val_logits(best_model, val_loader, device)
        temp_scaler = TemperatureScaler(num_classes=val_logits.shape[1])
        cal_stats = temp_scaler.fit_calibration(val_logits, val_targets)
        cal_probs = torch.sigmoid(temp_scaler(val_logits)).numpy()
        y_val_np = val_targets.numpy()
        opt_f1_thrs = f1_optimal_thresholds(y_val_np, cal_probs, beta=1.0)
        opt_youden_thrs = youden_thresholds(y_val_np, cal_probs)
        best_model.save(
            ckpt_dir / "best.pt",
            extra={
                "val_mean_auc": best_auc,
                "history": history,
                "temperatures": temp_scaler.temperature.detach().tolist(),
                "f1_thresholds": opt_f1_thrs,
                "youden_thresholds": opt_youden_thrs,
                "calibration": cal_stats,
            },
        )
        print(
            f"Fitted post-hoc per-class temperature calibration "
            f"(NLL {cal_stats['nll_before']:.4f} -> {cal_stats['nll_after']:.4f})"
        )

    print(f"\nBest val mean ROC-AUC: {best_auc:.4f}")
    return model
