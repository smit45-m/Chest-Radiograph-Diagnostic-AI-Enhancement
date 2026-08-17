"""Tests for Grad-CAM heatmap generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.interpret.gradcam import GradCAM, denormalize, overlay_cam, overlay_to_png_bytes


def test_gradcam_heatmap_shape(model, sample_image_tensor):
    """Heatmap should match input spatial dimensions."""
    with GradCAM(model, model.cam_target_layer()) as cam:
        heatmap = cam.generate(sample_image_tensor, class_idx=0)
    assert heatmap.shape == (224, 224)


def test_gradcam_heatmap_range(model, sample_image_tensor):
    """Heatmap values should be in [0, 1]."""
    with GradCAM(model, model.cam_target_layer()) as cam:
        heatmap = cam.generate(sample_image_tensor, class_idx=0)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_overlay_cam_shape():
    """Overlay should produce an RGB uint8 image."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(224, 224).astype(np.float32)
    result = overlay_cam(image, cam)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_overlay_to_png_bytes():
    """PNG bytes should start with the PNG magic number."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cam = np.random.rand(224, 224).astype(np.float32)
    png_data = overlay_to_png_bytes(image, cam)
    assert isinstance(png_data, bytes)
    assert png_data[:4] == b"\x89PNG"


def test_denormalize(sample_image_tensor):
    """Denormalized tensor should be an HxWx3 uint8 array."""
    result = denormalize(sample_image_tensor)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
