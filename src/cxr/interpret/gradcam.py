"""Grad-CAM for DenseNet-121.

We hook the last dense block, backprop a single class score, and weight
feature maps by the mean gradient. That produces a spatial heatmap of
what drove the pathology prediction.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cxr.constants import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._on_forward)
        # full_backward_hook is the non-deprecated hook that still sees the grad.
        self._bwd = target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _on_backward(self, _module, _gin, gout) -> None:
        self.gradients = gout[0].detach()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a [0, 1] heatmap of shape (H, W) matching the input image."""
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        # Grad-CAM needs activations + gradients, so we stay in inference
        # mode except for the backward of the chosen class score.
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        denom = cam.max() + 1e-8
        cam = cam / denom
        if was_training:
            self.model.train()
        return cam.astype(np.float32)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized 1x3xHxW tensor to an HxWx3 uint8 RGB image."""
    img = tensor.detach().cpu().squeeze(0).numpy()
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = (img * std + mean).clip(0, 1)
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Jet-colormap the CAM and blend it onto the CXR."""
    import cv2

    heat = np.uint8(np.clip(cam, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != heat.shape[:2]:
        heat = cv2.resize(heat, (image_rgb.shape[1], image_rgb.shape[0]))
    blended = (alpha * heat + (1 - alpha) * image_rgb).clip(0, 255).astype(np.uint8)
    return blended


def overlay_to_png_bytes(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> bytes:
    overlay = overlay_cam(image_rgb, cam, alpha=alpha)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()
