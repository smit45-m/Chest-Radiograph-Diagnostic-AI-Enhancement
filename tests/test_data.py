"""Tests for data pipeline utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.constants import PATHOLOGIES, PATHOLOGY_TO_INDEX
from cxr.data.labels import decode_findings, encode_findings, parse_finding_labels
from cxr.data.transforms import build_transforms


def test_encode_findings():
    """Multi-label string should produce correct multi-hot vector."""
    vec = encode_findings("Pneumonia|Effusion")
    assert vec.shape == (14,)
    assert vec[PATHOLOGY_TO_INDEX["Pneumonia"]] == 1.0
    assert vec[PATHOLOGY_TO_INDEX["Effusion"]] == 1.0
    assert vec.sum() == 2.0


def test_parse_finding_labels_no_finding():
    """'No Finding' should return an empty list."""
    assert parse_finding_labels("No Finding") == []
    assert parse_finding_labels("") == []
    assert parse_finding_labels(None) == []


def test_parse_finding_labels_multiple():
    """Pipe-separated labels should be parsed correctly."""
    result = parse_finding_labels("Cardiomegaly|Edema|Hernia")
    assert set(result) == {"Cardiomegaly", "Edema", "Hernia"}


def test_decode_findings():
    """Score vector should decode to (name, score) pairs above threshold."""
    scores = np.zeros(14, dtype=np.float32)
    scores[PATHOLOGY_TO_INDEX["Mass"]] = 0.8
    scores[PATHOLOGY_TO_INDEX["Nodule"]] = 0.3
    result = decode_findings(scores, threshold=0.5)
    assert len(result) == 1
    assert result[0][0] == "Mass"
    assert pytest.approx(result[0][1], abs=1e-5) == 0.8


def test_transforms_train_shape():
    """Train transforms should produce a (3, 224, 224) tensor."""
    transform = build_transforms(224, train=True)
    img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L")
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_transforms_eval_shape():
    """Eval transforms should produce a (3, 224, 224) tensor."""
    transform = build_transforms(224, train=False)
    img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L")
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
