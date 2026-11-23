"""Grad-CAM & Grad-CAM++ visual interpretability for thoracic radiology backbones.

Upgrades over vanilla Grad-CAM:
- `GradCAMPlusPlus` (Chattopadhay et al., WACV 2018): second- and third-order gradient
  pixel-wise weighting yields dramatically sharper multi-focal localization for bilateral
  infiltrates, multiple pulmonary nodules, and diffuse edema.
- Perceptually uniform medical colormaps (`turbo`, `inferno`, `viridis`, `clinical`, `jet`).
- Transparent RGBA heatmap PNG generation for 60 FPS client-side opacity & curtain blending.
- Automated Hotspot Bounding Box + Peak Coordinate + Anatomical Zone Localization
  intersected with PSPNet lung/heart segmentation masks.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cxr.constants import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Standard Grad-CAM + Grad-CAM++ engine (backward-compatible with existing tests)."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        method: str = "gradcam++",
    ) -> None:
        self.model = model
        self.method = method
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._on_forward)
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

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int,
        method: str | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a [0, 1] float32 heatmap of shape (H, W) matching input or `output_size`."""
        mode = (method or self.method).lower()
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        x_in = x.detach().clone().requires_grad_(True)
        use_amp = x_in.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = self.model(x_in)
            score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target layer.")

        acts = self.activations.float()
        grads = self.gradients.float()

        if mode in ("gradcam++", "plus", "pp"):
            # Grad-CAM++ closed-form pixel weights (Chattopadhay et al., WACV 2018)
            grads_sq = grads.pow(2)
            grads_cube = grads.pow(3)
            sum_acts = acts.sum(dim=(2, 3), keepdim=True)
            denom = 2.0 * grads_sq + sum_acts * grads_cube + 1e-7
            alpha = grads_sq / denom
            positive_grads = F.relu(grads)
            weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            # Fallback to vanilla if positive gradients are all zero
            if float(cam.max().item()) <= 1e-8:
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        target_hw = output_size if output_size is not None else (x.shape[-2], x.shape[-1])
        cam = F.interpolate(cam, size=target_hw, mode="bicubic", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = np.maximum(cam_np, 0.0)
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        if was_training:
            self.model.train()
        return np.clip(cam_np, 0.0, 1.0).astype(np.float32)


class GradCAMPlusPlus(GradCAM):
    """Explicit alias defaulting to Grad-CAM++ localization."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        super().__init__(model, target_layer, method="gradcam++")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized 1x3xHxW tensor to an HxWx3 uint8 RGB image."""
    img = tensor.detach().cpu().squeeze(0).numpy()
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = (img * std + mean).clip(0, 1)
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def _apply_colormap_rgb(cam: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Apply a perceptually uniform medical colormap and return (H, W, 3) uint8 RGB."""
    import cv2

    heat_u8 = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    cmap_key = (colormap or "turbo").lower()

    if cmap_key == "clinical":
        # Custom radiological gradient: transparent obsidian -> cyan -> amber -> crimson
        c = np.clip(cam, 0.0, 1.0)
        r = np.clip((c - 0.35) * 2.2, 0.0, 1.0)
        g = np.where(c < 0.55, np.clip(c * 1.5, 0.0, 0.85), np.clip((1.0 - c) * 1.8, 0.0, 0.85))
        b = np.clip((0.65 - c) * 1.6, 0.0, 1.0) * np.clip(c * 4.0, 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    cv_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
    }.get(cmap_key, cv2.COLORMAP_TURBO)

    colored = cv2.applyColorMap(heat_u8, cv_map)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_cam(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.48,
    colormap: str = "turbo",
) -> np.ndarray:
    """Colormap the CAM and blend smoothly onto the radiograph using activation-weighted alpha."""
    import cv2

    if image_rgb.shape[:2] != cam.shape[:2]:
        cam = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)
    heat = _apply_colormap_rgb(cam, colormap=colormap)

    # Activation-weighted alpha preserves unactivated dark lung fields cleanly
    local_alpha = (alpha * np.power(cam, 0.75))[..., None]
    blended = (local_alpha * heat + (1.0 - local_alpha) * image_rgb).clip(0, 255).astype(np.uint8)
    return blended


def overlay_to_png_bytes(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.48,
    colormap: str = "turbo",
) -> bytes:
    import cv2

    overlay = overlay_cam(image_rgb, cam, alpha=alpha, colormap=colormap)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok:
        return encoded.tobytes()
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


def cam_to_transparent_rgba_png(
    cam: np.ndarray,
    colormap: str = "turbo",
    threshold: float = 0.14,
) -> bytes:
    """Encode a pure transparent RGBA heatmap PNG for dynamic client-side blending."""
    import cv2

    cam_c = np.clip(cam, 0.0, 1.0)
    rgb = _apply_colormap_rgb(cam_c, colormap=colormap)
    scaled = np.maximum(0.0, cam_c - threshold) / (1.0 - threshold + 1e-6)
    alpha = np.where(cam_c >= threshold, np.power(scaled, 0.78), 0.0)
    rgba = np.dstack([rgb, np.uint8(np.clip(alpha * 235.0, 0, 255))])
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if ok:
        return encoded.tobytes()
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def extract_hotspot_metadata(
    cam: np.ndarray,
    seg_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute normalized hotspot peak, tight bounding box, coverage, and anatomical zone.

    Note radiological convention: image-left corresponds to the patient's RIGHT hemithorax,
    and image-right corresponds to the patient's LEFT hemithorax.
    """
    import cv2

    h, w = cam.shape[:2]
    cam_c = np.clip(cam, 0.0, 1.0)
    peak_idx = int(np.argmax(cam_c))
    py, px = divmod(peak_idx, w)
    peak_x = round(float(px) / max(w - 1, 1), 4)
    peak_y = round(float(py) / max(h - 1, 1), 4)
    peak_val = round(float(cam_c[py, px]), 4)

    hot_mask = cam_c >= max(0.55, peak_val * 0.62)
    ys, xs = np.where(hot_mask)
    if len(xs) > 0:
        x1 = round(float(xs.min()) / max(w - 1, 1), 4)
        y1 = round(float(ys.min()) / max(h - 1, 1), 4)
        x2 = round(float(xs.max()) / max(w - 1, 1), 4)
        y2 = round(float(ys.max()) / max(h - 1, 1), 4)
        coverage = round(float(hot_mask.mean()) * 100.0, 2)
    else:
        x1, y1, x2, y2, coverage = 0.25, 0.25, 0.75, 0.75, 0.0

    # Determine anatomical zone via segmentation masks if available, else thoracic grid
    zone = _infer_anatomical_zone(cam_c, peak_x, peak_y, seg_masks)

    # Also check bilateral involvement
    left_energy = float(cam_c[:, : w // 2].mean())   # Patient's Right Lung
    right_energy = float(cam_c[:, w // 2 :].mean())  # Patient's Left Lung
    bilateral = min(left_energy, right_energy) / max(left_energy, right_energy, 1e-6) > 0.68 and coverage > 10.0

    return {
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_intensity": peak_val,
        "bbox": [x1, y1, x2, y2],
        "coverage_pct": coverage,
        "anatomical_zone": zone,
        "laterality": "Bilateral" if bilateral else ("Right Hemithorax" if peak_x < 0.45 else ("Left Hemithorax" if peak_x > 0.55 else "Midline / Cardiomediastinal")),
    }


def _infer_anatomical_zone(
    cam: np.ndarray,
    peak_x: float,
    peak_y: float,
    seg_masks: dict[str, np.ndarray] | None,
) -> str:
    import cv2

    if seg_masks:
        h, w = cam.shape[:2]
        hot = cam >= 0.50
        overlaps: dict[str, float] = {}
        for label in ("Right Lung", "Left Lung", "Heart", "Mediastinum", "Aorta"):
            mask = seg_masks.get(label)
            if mask is None:
                continue
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            area = float(mask.sum())
            if area > 10:
                overlaps[label] = float((hot & mask).sum()) / max(float(hot.sum()), 1.0)

        if overlaps:
            # Prioritize Cardiomediastinal Silhouette when peak overlaps heart/mediastinum strongly
            heart_ov = overlaps.get("Heart", 0.0) + overlaps.get("Mediastinum", 0.0) * 0.8
            if heart_ov >= 0.24:
                return "Cardiomediastinal Silhouette"
            best_organ, best_ov = max(overlaps.items(), key=lambda kv: kv[1])
            if best_ov >= 0.22:
                if best_organ in ("Heart", "Mediastinum", "Aorta"):
                    return "Cardiomediastinal Silhouette"
                zone_v = "Upper Zone" if peak_y < 0.36 else ("Mid Zone" if peak_y < 0.62 else "Lower / Basilar Zone")
                return f"{best_organ} — {zone_v}"

    # Radiological fallback grid (image-left = patient Right)
    if 0.38 <= peak_x <= 0.62 and 0.42 <= peak_y <= 0.80:
        return "Cardiomediastinal Silhouette"
    side = "Right Lung" if peak_x < 0.5 else "Left Lung"
    if peak_y < 0.35:
        level = "Upper / Apical Zone"
    elif peak_y < 0.62:
        level = "Mid Zone"
    else:
        level = "Lower / Costophrenic Zone"
    return f"{side} — {level}"
