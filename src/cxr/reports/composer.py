"""Deterministic Findings / Impression draft from the 14-class score vector.

This is the always-on first stage. A local T5 or SpaceXAI Grok can polish
the draft afterwards. The draft is already a valid radiology-style note,
so the pipeline works with no API key and no extra model download.
"""

from __future__ import annotations

from cxr.constants import FINDING_LANGUAGE, PATHOLOGIES


def _severity(score: float) -> str:
    if score >= 0.85:
        return "severe"
    if score >= 0.65:
        return "moderate"
    return "mild"


def compose_report(
    scores: dict[str, float],
    threshold: float = 0.45,
    image_id: str | None = None,
) -> dict:
    positives = [(name, scores[name]) for name in PATHOLOGIES if scores.get(name, 0.0) >= threshold]
    positives.sort(key=lambda x: x[1], reverse=True)

    if not positives:
        findings = (
            "The lungs are clear without focal consolidation, effusion, or pneumothorax. "
            "The cardiomediastinal silhouette is within normal limits. "
            "No acute osseous abnormality is identified."
        )
        impression = "No acute cardiopulmonary process."
        technique = "Frontal chest radiograph."
    else:
        sentences = []
        for name, score in positives:
            band = _severity(score)
            phrase = FINDING_LANGUAGE[name][band]
            sentences.append(f"{phrase.capitalize()} (model score {score:.2f}).")
        findings = " ".join(sentences)
        names = ", ".join(name.replace("_", " ") for name, _ in positives)
        if len(positives) == 1:
            impression = f"Findings most suggestive of {names}. Clinical correlation is recommended."
        else:
            impression = (
                f"Multi-pathology study. Leading considerations: {names}. "
                "Correlate with symptoms, labs, and prior imaging."
            )
        technique = "Frontal chest radiograph. Automated multi-label interpretation."

    header = "EXAMINATION: CHEST RADIOGRAPH (FRONTAL)"
    if image_id:
        header += f"\nACCESSION: {image_id}"

    draft = (
        f"{header}\n"
        f"TECHNIQUE: {technique}\n\n"
        f"FINDINGS:\n{findings}\n\n"
        f"IMPRESSION:\n{impression}\n\n"
        "DISCLAIMER: Research prototype. Not for clinical use. "
        "A licensed radiologist must review the images."
    )
    return {
        "findings": findings,
        "impression": impression,
        "positives": [{"pathology": n, "score": float(s), "severity": _severity(s)} for n, s in positives],
        "draft": draft,
        "source": "template",
    }
