"""Tests for DenseNet121Classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.constants import NUM_CLASSES
from cxr.models.densenet import DenseNet121Classifier


def test_forward_shape(model, sample_image_tensor):
    """Model output has shape (B, 14)."""
    out = model(sample_image_tensor)
    assert out.shape == (1, NUM_CLASSES)


def test_output_range(model, sample_image_tensor):
    """Sigmoid of logits should be in [0, 1]."""
    logits = model(sample_image_tensor)
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_save_load_roundtrip(model, sample_image_tensor, tmp_path):
    """Save → load → forward should produce identical outputs."""
    ckpt = tmp_path / "model.pt"
    model.eval()
    model.save(ckpt)

    loaded = DenseNet121Classifier.load(ckpt)
    loaded.eval()

    with torch.no_grad():
        out1 = model(sample_image_tensor)
        out2 = loaded(sample_image_tensor)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_freeze_unfreeze(model):
    """freeze_backbone / unfreeze_backbone toggles requires_grad."""
    model.freeze_backbone()
    backbone_params = [
        p for n, p in model.backbone.named_parameters()
        if not n.startswith("classifier")
    ]
    assert all(not p.requires_grad for p in backbone_params)

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.parameters())


def test_cam_target_layer(model):
    """cam_target_layer() returns a valid nn.Module."""
    layer = model.cam_target_layer()
    assert isinstance(layer, torch.nn.Module)
