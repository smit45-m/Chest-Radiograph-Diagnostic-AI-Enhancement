"""SpaceXAI (xAI Grok) polisher for the radiology draft.

Uses the OpenAI-compatible API at https://api.x.ai/v1. If XAI_API_KEY is
missing, callers should skip this module and keep the template draft.
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = """You are a cautious radiology reporting assistant.
Rewrite the structured draft into a concise, professional chest radiograph report.

Rules:
- Keep FINDINGS and IMPRESSION as separate sections.
- Do not invent anatomy, laterality, measurements, or comparison studies that are not in the draft.
- Hedge appropriately: these scores come from a research model, not a radiologist.
- End with a one-line reminder that a licensed physician must sign the report.
- Do not mention being an AI unless asked.
"""


def polish_with_grok(draft: str, model: str | None = None) -> str:
    from openai import OpenAI

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    chosen = model or os.environ.get("CXR_LLM_MODEL", "grok-4.6")
    resp = client.chat.completions.create(
        model=chosen,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": draft},
        ],
    )
    text = resp.choices[0].message.content or ""
    return text.strip()
