"""Class-balanced binary cross-entropy for 14-way multi-label training."""

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


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
