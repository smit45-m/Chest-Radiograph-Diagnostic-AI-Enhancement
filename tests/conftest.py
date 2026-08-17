"""Shared pytest fixtures for the CXR test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.constants import PATHOLOGIES  # noqa: E402
from cxr.models.densenet import DenseNet121Classifier  # noqa: E402


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_image_tensor() -> torch.Tensor:
    """Random 1×3×224×224 tensor simulating a preprocessed CXR."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_scores() -> dict[str, float]:
    """Random scores for all 14 pathologies."""
    import random
    random.seed(42)
    return {name: round(random.random(), 3) for name in PATHOLOGIES}


@pytest.fixture
def model() -> DenseNet121Classifier:
    """Lightweight model with random weights (no ImageNet download)."""
    return DenseNet121Classifier(pretrained="none", dropout=0.0)


@pytest.fixture
def tmp_checkpoint(model, tmp_path) -> Path:
    """Save a model checkpoint to a temporary directory."""
    ckpt_path = tmp_path / "test_model.pt"
    model.save(ckpt_path)
    return ckpt_path
