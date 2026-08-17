"""Multi-label encoding for NIH Finding Labels strings."""

from __future__ import annotations

import numpy as np

from cxr.constants import LABEL_ALIASES, NUM_CLASSES, PATHOLOGY_TO_INDEX, PATHOLOGIES


def parse_finding_labels(raw: str) -> list[str]:
    """Split a NIH 'Finding Labels' cell into canonical pathology names."""
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "No Finding":
        return []
    names: list[str] = []
    for token in raw.split("|"):
        token = token.strip()
        token = LABEL_ALIASES.get(token, token)
        if token in PATHOLOGY_TO_INDEX:
            names.append(token)
    return names


def encode_findings(raw: str) -> np.ndarray:
    """Return a float32 multi-hot vector of shape (14,)."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for name in parse_finding_labels(raw):
        vec[PATHOLOGY_TO_INDEX[name]] = 1.0
    return vec


def decode_findings(vec, threshold: float = 0.5) -> list[tuple[str, float]]:
    """Turn a score/label vector into (name, score) pairs above threshold."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    hits = []
    for i, name in enumerate(PATHOLOGIES):
        if float(vec[i]) >= threshold:
            hits.append((name, float(vec[i])))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits
