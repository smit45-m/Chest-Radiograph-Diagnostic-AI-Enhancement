"""Loss functions for imbalanced multi-label thoracic pathology classification.

Provides:
- Class-balanced Binary Cross-Entropy (`pos_weight_from_labels`, `build_loss`)
- Multi-Label Asymmetric Loss (`AsymmetricLossMultiLabel`, Ridnik et al., ICCV 2021)
  which decouples positive/negative focusing and probability margin shifting to
  excel on extreme class imbalance (e.g. NIH ChestX-ray14 Hernia prevalence 0.16%).
- Multi-Label Focal Loss (`MultiLabelFocalLoss`)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pos_weight_from_labels(labels: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    """pos_weight_c = (N - pos_c) / pos_c, clipped so Hernia does not explode."""
    pos = labels.sum(axis=0).astype(np.float64)
    n = float(len(labels))
    pos = np.clip(pos, 1.0, None)
    weight = (n - pos) / pos
    weight = np.clip(weight, 1.0, max_weight)
    return torch.tensor(weight, dtype=torch.float32)


class AsymmetricLossMultiLabel(nn.Module):
    """Asymmetric Loss (ASL) for severely imbalanced multi-label classification.

    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    Down-weights easy negatives aggressively (gamma_neg > gamma_pos) and applies
    asymmetric probability margin clipping (`clip`) to suppress mislabeled/noisy negatives
    common in NLP-mined radiology labels (NIH ChestX-ray14 / CheXpert).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """Multi-label sigmoid focal loss with optional positive class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()


def build_loss(
    pos_weight: torch.Tensor | None = None,
    loss_type: str = "bce",
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Create a multi-label loss function (`bce`, `asl`, or `focal`)."""
    kind = (loss_type or "bce").lower()
    if kind == "asl":
        return AsymmetricLossMultiLabel(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            label_smoothing=label_smoothing,
        )
    if kind == "focal":
        return MultiLabelFocalLoss(gamma=2.0, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
