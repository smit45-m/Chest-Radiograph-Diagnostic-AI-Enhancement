"""Three-stage report pipeline: template → optional T5 → optional Grok."""

from __future__ import annotations

from pathlib import Path

from cxr.reports.composer import compose_report


def generate_report(
    scores: dict[str, float],
    threshold: float = 0.45,
    image_id: str | None = None,
    use_t5: bool = True,
    use_llm: bool = True,
    t5_dir: str | Path = "checkpoints/report_t5",
) -> dict:
    report = compose_report(scores, threshold=threshold, image_id=image_id)
    text = report["draft"]
    stage = "template"

    if use_t5:
        try:
            from cxr.reports.transformer import rewrite_with_t5

            text = rewrite_with_t5(text, ckpt_dir=t5_dir)
            stage = "t5"
        except Exception:
            pass

    if use_llm:
        try:
            from cxr.reports.llm import polish_with_grok

            text = polish_with_grok(text if stage != "template" else report["draft"])
            stage = "llm"
        except Exception:
            pass

    report["report"] = text
    report["stage"] = stage
    return report
