"""Transfer learning thoracic pathology classifiers & post-hoc calibration.

Provides:
- `DenseNet121Classifier`: CheXNet-style DenseNet-121 with optional Generalized Mean
  (GeM) pooling and Multi-Sample Dropout head (fully backward-compatible with v1 ckpts).
- `ThoracicTransferModel`: Multi-architecture transfer-learning factory supporting
  `densenet121`, `convnext_small`, `efficientnet_v2_s`, and `resnet50` with layer-wise
  learning rate decay (LLRD) parameter grouping and Grad-CAM++ target layer hooks.
- `TemperatureScaler`: Per-class post-hoc temperature scaling to minimize Expected
  Calibration Error (ECE) on multi-label clinical probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Small_Weights,
    DenseNet121_Weights,
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
    convnext_small,
    densenet121,
    efficientnet_v2_s,
    resnet50,
)

from cxr.constants import NUM_CLASSES


class GeM2d(nn.Module):
    """Generalized Mean (GeM) spatial pooling for fine-grained lesion localization.

    Interpolates smoothly between average pooling (p=1) and max pooling (p→∞),
    capturing focal nodules/pneumothorax much better than global average pooling alone.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class MultiSampleDropoutHead(nn.Module):
    """Multi-sample dropout classifier averaging 5 stochastic masks (Inoue, 2019)."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.25, samples: int = 5) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(samples)])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.fc(x)
        return torch.stack([self.fc(d(x)) for d in self.dropouts], dim=0).mean(dim=0)


class DenseNet121Classifier(nn.Module):
    """ImageNet-initialized DenseNet-121 with a 14-way sigmoid head.

    Keeps the torchvision backbone intact so Grad-CAM / Grad-CAM++ can hook
    `backbone.features.denseblock4` (last dense block) cleanly.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: str = "imagenet",
        dropout: float = 0.2,
        multi_sample_dropout: bool = False,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        if multi_sample_dropout:
            self.backbone.classifier = MultiSampleDropoutHead(
                in_features, num_classes, dropout=dropout
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        self.num_classes = num_classes
        self.multi_sample_dropout = multi_sample_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def cam_target_layer(self) -> nn.Module:
        """Last convolutional block — standard Grad-CAM / Grad-CAM++ target for DenseNet."""
        return self.backbone.features.denseblock4

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def param_groups_llrd(self, base_lr: float, weight_decay: float = 1e-5) -> list[dict[str, Any]]:
        """Layer-wise learning rate decay: classifier 1.0x, late dense blocks 0.3x, stem 0.1x."""
        stem_params, mid_params, late_params, head_params = [], [], [], []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(param)
            elif "denseblock4" in name or "norm5" in name:
                late_params.append(param)
            elif "denseblock3" in name or "transition3" in name:
                mid_params.append(param)
            else:
                stem_params.append(param)
        groups = []
        for params, mult in (
            (stem_params, 0.1),
            (mid_params, 0.3),
            (late_params, 0.6),
            (head_params, 1.0),
        ):
            if params:
                groups.append({"params": params, "lr": base_lr * mult, "weight_decay": weight_decay})
        return groups

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "multi_sample_dropout": self.multi_sample_dropout,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "DenseNet121Classifier":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            num_classes=int(ckpt.get("num_classes", NUM_CLASSES)),
            pretrained="none",
            multi_sample_dropout=bool(ckpt.get("multi_sample_dropout", False)),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


class ThoracicTransferModel(nn.Module):
    """Unified transfer learning wrapper for modern thoracic imaging backbones.

    Supported architectures:
    - `densenet121` (classic CheXNet)
    - `convnext_small` (modern depthwise ConvNeXt with large 7x7 kernels)
    - `efficientnet_v2_s` (high-efficiency compound scaling)
    - `resnet50` (high-resolution spatial residual features)
    """

    def __init__(
        self,
        arch: str = "densenet121",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.25,
        use_gem: bool = True,
    ) -> None:
        super().__init__()
        self.arch = arch.lower()
        self.num_classes = num_classes
        self.use_gem = use_gem

        if self.arch == "densenet121":
            m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = m.features
            in_features = m.classifier.in_features
            self._cam_layer = self.features.denseblock4
        elif self.arch == "convnext_small":
            m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = m.features
            in_features = m.classifier[2].in_features
            self._cam_layer = self.features[-1]
        elif self.arch == "efficientnet_v2_s":
            m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = m.features
            in_features = m.classifier[1].in_features
            self._cam_layer = self.features[-1]
        elif self.arch == "resnet50":
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.features = nn.Sequential(
                m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4
            )
            in_features = m.fc.in_features
            self._cam_layer = self.features[-1]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.pool = GeM2d(p=3.0) if use_gem else nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(in_features)
        self.head = MultiSampleDropoutHead(in_features, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        if self.arch == "densenet121":
            feat = F.relu(feat, inplace=False)
        pooled = self.pool(feat).flatten(1)
        return self.head(self.norm(pooled))

    def cam_target_layer(self) -> nn.Module:
        return self._cam_layer

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True


class TemperatureScaler(nn.Module):
    """Per-class post-hoc temperature scaling for calibrated clinical probabilities."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_classes))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.25, max=5.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit_calibration(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        max_iter: int = 80,
    ) -> dict[str, float]:
        """Fit per-class temperatures via LBFGS minimizing negative log-likelihood."""
        val_logits = val_logits.detach().float()
        val_targets = val_targets.detach().float()
        criterion = nn.BCEWithLogitsLoss()
        before = float(criterion(val_logits, val_targets).item())
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        after = float(criterion(self.forward(val_logits), val_targets).item())
        return {"nll_before": round(before, 5), "nll_after": round(after, 5)}
