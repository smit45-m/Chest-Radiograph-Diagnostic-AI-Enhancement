"""Clinical LLM Polisher & Radiologist Report Synthesis.

Supports:
1. Groq Cloud (Free Tier - ultra-fast Llama 3.3 70B / Llama 3.1 8B)
2. SpaceXAI / xAI Grok
3. OpenAI / Custom OpenAI-compatible endpoints
4. Built-in Advanced Clinical NLP Radiologist Synthesizer (Zero-dependency fallback)
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = """You are an expert board-certified thoracic radiologist dictating a formal chest radiograph report.
Based on the provided automated multi-label AI predictions and initial clinical findings draft, generate an authoritative, clear, and standardized clinical radiology report.

Format the output strictly as:
EXAMINATION: CHEST RADIOGRAPH (FRONTAL)
INDICATION: Automated Multi-Pathology Diagnostic Evaluation
TECHNIQUE: Single view frontal chest radiograph (PA/AP)

FINDINGS:
- Detailed descriptive paragraph covering Lungs & Pleural Spaces, Cardiomediastinal Contour, and Osseous Structures / Soft Tissues.

IMPRESSION:
- Numbered, prioritized diagnostic conclusions based strictly on the detected pathologies.
- Differential diagnoses where applicable.

RECOMMENDATIONS:
- Clear clinical next steps (e.g. correlative imaging, repeat radiograph, clinical biomarker correlation).

DISCLAIMER:
Automated AI diagnostic synthesis for investigational use. Findings must be validated by a licensed physician before clinical decision-making.
"""


def _synthesize_clinical_report(draft: str) -> str:
    """Advanced rule-guided Clinical NLP synthesizer when no external API key is present."""
    # Extract findings if present
    lines = [l.strip() for l in draft.split("\n") if l.strip()]
    findings_lines = []
    impression_lines = []
    mode = None

    for line in lines:
        if "FINDINGS:" in line:
            mode = "findings"
            continue
        elif "IMPRESSION:" in line:
            mode = "impression"
            continue
        elif "DISCLAIMER:" in line or "TECHNIQUE:" in line:
            mode = None
            continue

        if mode == "findings":
            findings_lines.append(line)
        elif mode == "impression":
            impression_lines.append(line)

    findings_text = " ".join(findings_lines) if findings_lines else "The lungs demonstrate clear parenchyma without focal consolidation, pneumothorax, or pleural effusion. The cardiomediastinal contour is preserved."
    impression_text = "\n".join(f"1. {l}" for l in impression_lines) if impression_lines else "1. No acute cardiopulmonary abnormality identified."

    # Build recommendations based on keywords
    recs = []
    lower = draft.lower()
    if "cardiomegaly" in lower or "edema" in lower:
        recs.append("1. Correlate with clinical volume status, NT-proBNP, and consider 2D-Echocardiogram to assess left ventricular ejection fraction.")
    if "pneumothorax" in lower:
        recs.append("2. Urgent clinical evaluation. Consider upright expiratory view or thoracic CT if tension physiology is suspected.")
    if "pneumonia" in lower or "consolidation" in lower or "infiltrate" in lower:
        recs.append("3. Targeted antimicrobial therapy as clinically indicated. Recommend interval follow-up radiograph in 4-6 weeks to confirm resolution.")
    if "effusion" in lower:
        recs.append("4. Consider diagnostic thoracentesis or lateral decubitus radiograph if clinically indicated for pleural fluid characterization.")
    if not recs:
        recs.append("1. Routine clinical follow-up as indicated. No emergent thoracic intervention required.")

    recs_text = "\n".join(recs)

    return f"""EXAMINATION: CHEST RADIOGRAPH (FRONTAL)
TECHNIQUE: Automated DenseNet-121 Multi-Label Interpretation & Deep Feature Localization.

FINDINGS:
{findings_text}
Bilateral thoracic cage, visualised ribs, and diaphragmatic contours demonstrate no acute fractures or gross structural disruption.

IMPRESSION:
{impression_text}

RECOMMENDATIONS:
{recs_text}

DISCLAIMER:
Research prototype for investigational use. All findings and synthesized dictations must be reviewed and countersigned by a licensed physician."""


def polish_with_llm(
    draft: str,
    api_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Polish a structured CXR draft using Groq, xAI, OpenAI, or the built-in clinical synthesizer."""
    groq_key = api_key or os.environ.get("GROQ_API_KEY")
    xai_key = api_key or os.environ.get("XAI_API_KEY")
    openai_key = api_key or os.environ.get("OPENAI_API_KEY")

    # 1. Try Groq (Free Tier, Fast Llama 3.3 70B / 3.1 8B)
    if groq_key and (provider == "groq" or not provider):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            chosen_model = model or os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
            resp = client.chat.completions.create(
                model=chosen_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a polished, formal clinical radiology report based on this preliminary AI diagnostic draft:\n\n{draft}"},
                ],
                max_tokens=600,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"[llm] Groq call failed ({exc}); trying fallback...")

    # 2. Try xAI Grok
    if xai_key and (provider == "xai" or not provider):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
            chosen_model = model or os.environ.get("CXR_LLM_MODEL", "grok-4.6")
            resp = client.chat.completions.create(
                model=chosen_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Polish this draft radiology report:\n\n{draft}"},
                ],
                max_tokens=600,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"[llm] xAI Grok call failed ({exc}); trying fallback...")

    # 3. Try Standard OpenAI
    if openai_key and (provider == "openai" or not provider):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            chosen_model = model or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=chosen_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a clinical chest X-ray report from this draft:\n\n{draft}"},
                ],
                max_tokens=600,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"[llm] OpenAI call failed ({exc}); using built-in synthesizer...")

    # 4. High-Yield Built-in Clinical NLP Synthesizer (Always Succeeds)
    return _synthesize_clinical_report(draft)


def polish_with_grok(draft: str, model: str | None = None) -> str:
    """Backwards-compatible alias for polish_with_llm."""
    return polish_with_llm(draft, model=model)
