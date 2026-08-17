"""DenseNet-121 multi-label classifier (CheXNet-style)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from cxr.constants import NUM_CLASSES


class DenseNet121Classifier(nn.Module):
    """ImageNet-initialized DenseNet-121 with a 14-way sigmoid head.

    We keep the torchvision backbone intact so Grad-CAM can hook
    `backbone.features.denseblock4` (last dense block) cleanly.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: str = "imagenet",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def cam_target_layer(self) -> nn.Module:
        """Last convolutional block — the standard Grad-CAM target for DenseNet."""
        return self.backbone.features.denseblock4

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "DenseNet121Classifier":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(num_classes=int(ckpt.get("num_classes", NUM_CLASSES)), pretrained="none")
        model.load_state_dict(ckpt["state_dict"])
        return model
