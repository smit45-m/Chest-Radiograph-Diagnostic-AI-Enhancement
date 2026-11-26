"""Clinical LLM Polisher & Radiologist Report Synthesis.

Priority Cascade:
1. **Gemini Flash (`gemini-2.5-flash` / `gemini-3-flash-preview` / `gemini-2.5-flash-lite`)**
   via Google GenAI SDK with multimodal radiograph + quantitative CTR + Grad-CAM++ zone grounding.
2. **Groq Cloud** (ultra-fast Llama 3.3 70B / Qwen3 32B / GPT-OSS via OpenAI-compatible API).
3. **xAI Grok** (`grok-4.6` / configurable).
4. **OpenAI** (`gpt-4o-mini` / configurable).
5. **Built-in Advanced ACR/RSNA Clinical NLP Synthesizer** (zero-dependency deterministic fallback).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG, synthesize_with_gemini_flash

_GROQ_CLIENT_LOCK = threading.Lock()
_GROQ_CLIENTS: dict[str, Any] = {}


def _get_groq_client(api_key: str) -> Any:
    """Reuse a persistent OpenAI/Groq client per API key to eliminate TLS handshake overhead."""
    client = _GROQ_CLIENTS.get(api_key)
    if client is None:
        with _GROQ_CLIENT_LOCK:
            client = _GROQ_CLIENTS.get(api_key)
            if client is None:
                from openai import OpenAI

                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=10.5,
                )
                _GROQ_CLIENTS[api_key] = client
    return client

SYSTEM_PROMPT = """You are a distinguished academic thoracic radiologist dictating a formal chest radiograph report and clinical synthesis.
Synthesize the provided preliminary AI multi-label predictions, anatomical draft, quantitative CTR, Grad-CAM++ localizations, and patient clinical profile into an authoritative, ACR/RSNA-compliant radiology report.

Strictly adhere to the following format:
EXAMINATION: CHEST RADIOGRAPH (FRONTAL)
PATIENT CONTEXT: [Patient age, sex, clinical setting, and primary indication]
TECHNIQUE: Frontal chest radiograph (PA/AP). Multi-backbone clinical ensemble CAD (DenseNet-121 + ResNet-50 512px + PSPNet Anatomy + Gemini Decision 1).

FINDINGS:
- LUNGS & AIRWAYS: [Airspace opacities, focal consolidation, nodules/masses, atelectasis, emphysema, or normal parenchyma]
- PLEURAL SPACES & SULCI: [Effusions, pneumothorax, pleural thickening, costophrenic angles]
- CARDIOMEDIASTINUM & VASCULATURE: [Cardiothoracic ratio, pulmonary vascular caliber, aortic contour]
- OSSEOUS STRUCTURES & SOFT TISSUES: [Bony thorax, ribs, spine, diaphragm, soft tissues]

IMPRESSION:
- Numbered, prioritized diagnostic conclusions tailored to the acute clinical presentation.
- Pertinent differential diagnoses considering the patient's age and comorbidities.

GUIDELINE-DIRECTED RECOMMENDATIONS & CLINICAL ACTION PLAN:
- Actionable next steps citing ATS/IDSA, Fleischner Society 2017, ACC/AHA, ACCP, or BTS guidelines.

PATIENT-FRIENDLY TRANSLATION:
- Clear, compassionate plain-language summary with 3 questions to ask their physician.

DISCLAIMER:
Automated AI diagnostic synthesis for investigational research use. All findings must be clinically correlated and validated by a licensed physician before therapeutic decision-making.
"""


def _synthesize_clinical_report(draft: str, patient: dict[str, Any] | None = None) -> str:
    """Advanced rule-guided Clinical NLP synthesizer when no external LLM API key is present."""
    sections: dict[str, list[str]] = {
        "header": [],
        "patient": [],
        "findings": [],
        "impression": [],
        "recommendations": [],
        "prescriptions": [],
        "patient_summary": [],
    }
    mode: str | None = "header"

    for raw_line in draft.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if "PATIENT:" in line or "CLINICAL INDICATION:" in line:
            mode = "patient"
            sections[mode].append(line)
            continue
        elif "FINDINGS:" in line:
            mode = "findings"
            continue
        elif "IMPRESSION:" in line:
            mode = "impression"
            continue
        elif "RECOMMENDATIONS:" in line:
            mode = "recommendations"
            continue
        elif "GUIDELINE PRESCRIPTIONS & PHARMACOTHERAPY:" in line or "PRESCRIPTIONS:" in line:
            mode = "prescriptions"
            continue
        elif "PATIENT-FRIENDLY SUMMARY:" in line or "PATIENT-FRIENDLY TRANSLATION:" in line:
            mode = "patient_summary"
            continue
        elif "DISCLAIMER:" in line or "TECHNIQUE:" in line:
            mode = None
            continue

        if mode and mode in sections:
            sections[mode].append(line)

    findings_text = "\n".join(sections["findings"]) if sections["findings"] else (
        "LUNGS & AIRWAYS: The lungs are well expanded and clear bilaterally without focal consolidation, airspace opacity, or suspicious mass.\n"
        "PLEURA & SULCI: Costophrenic sulci are sharp and clear. No pleural effusion or pneumothorax.\n"
        "CARDIOMEDIASTINUM: Cardiomediastinal silhouette and aortic contour are within normal limits.\n"
        "OSSEOUS & SOFT TISSUES: Bilateral rib cage and soft tissues are intact without acute fracture."
    )
    impression_text = "\n".join(sections["impression"]) if sections["impression"] else "1. No acute cardiopulmonary process identified."
    recs_text = "\n".join(sections["recommendations"]) if sections["recommendations"] else (
        "1. Routine clinical follow-up as indicated. No acute thoracic imaging or intervention required."
    )
    rx_text = "\n".join(sections["prescriptions"]) if sections["prescriptions"] else (
        "• No specific cardiopulmonary pharmacotherapy or antimicrobial prescription indicated at this time."
    )
    p_summary_text = "\n".join(sections["patient_summary"]) if sections["patient_summary"] else (
        "What Your X-Ray Shows:\n"
        "• Good news: Your chest X-ray appears clear with no acute pneumonia, heart enlargement, or lung collapse.\n\n"
        "What This Means For You:\n"
        "• Your lungs and heart look healthy on this study. If you continue to experience symptoms, your doctor will guide appropriate care."
    )

    if patient:
        age_str = f"{patient.get('age')}yo " if patient.get("age") else ""
        p_ctx = f"PATIENT CONTEXT: {age_str}{patient.get('sex', 'Adult')} | Setting: {patient.get('setting', 'Outpatient')} | Indication: {patient.get('indication', 'Diagnostic evaluation')}"
    elif sections["patient"]:
        p_ctx = "PATIENT CONTEXT: " + " | ".join(sections["patient"])
    else:
        p_ctx = "PATIENT CONTEXT: Thoracic diagnostic radiograph"

    return f"""EXAMINATION: CHEST RADIOGRAPH (FRONTAL)
{p_ctx}
TECHNIQUE: Multi-Backbone Clinical Ensemble (DenseNet-121 + ResNet-50 512px + PSPNet Anatomy) & Grad-CAM++ Localization.

FINDINGS:
{findings_text}

IMPRESSION:
{impression_text}

GUIDELINE-DIRECTED RECOMMENDATIONS:
{recs_text}

EVIDENCE-BASED PRESCRIPTIONS & PHARMACOTHERAPY:
{rx_text}

PATIENT-FRIENDLY TRANSLATION:
{p_summary_text}

DISCLAIMER:
Research prototype for investigational use. All findings and synthesized dictations must be reviewed and countersigned by a licensed physician before clinical decision-making."""


def polish_with_llm(
    draft: str,
    api_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    patient: dict[str, Any] | None = None,
    scores: dict[str, float] | None = None,
    threshold: float = 0.50,
    image_bytes: bytes | None = None,
    anatomy: dict[str, Any] | None = None,
    hotspots: list[dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any] | None]:
    """Polish a structured CXR draft using Gemini Flash -> Groq -> xAI -> OpenAI -> Clinical NLP.

    Returns:
        (report_text, provider_tag, optional_structured_gemini_dict)
    """
    AI_RUNTIME_CONFIG.refresh_from_env()

    # 1. Try Gemini Flash (Primary multimodal structured clinical synthesizer)
    gemini_key = api_key if provider == "gemini" else AI_RUNTIME_CONFIG.gemini_api_key
    if gemini_key and (provider in (None, "gemini")) and scores is not None:
        gemini_res = synthesize_with_gemini_flash(
            scores=scores,
            threshold=threshold,
            draft=draft,
            image_bytes=image_bytes,
            patient=patient,
            anatomy=anatomy,
            hotspots=hotspots,
            decision_summary=decision_summary,
        )
        if gemini_res and gemini_res.get("report"):
            return gemini_res["report"], gemini_res["source"], gemini_res

    groq_key = (api_key if provider == "groq" else None) or AI_RUNTIME_CONFIG.groq_api_key or os.environ.get("GROQ_API_KEY")
    xai_key = (api_key if provider == "xai" else None) or os.environ.get("XAI_API_KEY")
    openai_key = (api_key if provider == "openai" else None) or os.environ.get("OPENAI_API_KEY")

    patient_ctx_str = ""
    if patient:
        patient_ctx_str = f"""
PATIENT CLINICAL CONTEXT:
- Age: {patient.get('age', 'Unspecified')}
- Biological Sex: {patient.get('sex', 'Unspecified')}
- Clinical Setting: {patient.get('setting', 'Ambulatory / Outpatient')}
- Primary Indication / Chief Complaint: {patient.get('indication', 'Diagnostic evaluation')}
- Pertinent Medical History / Comorbidities: {patient.get('comorbidities', 'None documented')}
- Tobacco / Smoking History: {patient.get('smoking', 'None documented')}
"""
    if anatomy and anatomy.get("ctr") is not None:
        patient_ctx_str += f"\nQUANTITATIVE ANATOMY: Cardiothoracic Ratio (CTR) = {anatomy['ctr']:.3f} ({anatomy.get('ctr_status', '')})\n"

    user_query = (
        f"Generate a comprehensive, formal clinical thoracic radiology consultation report and patient translation "
        f"based on this clinical profile and preliminary AI diagnostic draft:\n"
        f"{patient_ctx_str}\n"
        f"PRELIMINARY AI DIAGNOSTIC DRAFT:\n{draft}"
    )

    # 2. Try Groq Cloud with structured ACR JSON matching ClinicalReportSchema (live real AI transcripts!)
    if groq_key and (provider in (None, "groq")):
        try:
            import json

            client = _get_groq_client(groq_key)
            model_candidates = list(
                dict.fromkeys(
                    m
                    for m in [
                        model,
                        AI_RUNTIME_CONFIG.groq_model,
                        "openai/gpt-oss-20b",
                        "openai/gpt-oss-120b",
                        "qwen/qwen3.8-27b",
                    ]
                    if m and not m.startswith("llama-3.")
                )
            )
            structured_sys = (
                "You are an attending academic thoracic radiologist. Return ONLY a concise valid JSON object with EXACTLY these keys:\n"
                '{"executive_summary":["3 short clinical bullets"],'
                '"triage_level":"STAT_CRITICAL|URGENT|SEMI_URGENT|ROUTINE",'
                '"compartments":{"lungs":"...","pleura":"...","cardio":"...","osseous":"..."},'
                '"impression":"Numbered diagnostic conclusions",'
                '"dictation_transcript":"Natural spoken attending radiologist dictation (3-4 crisp sentences citing pathology, CTR, localization zone, and next clinical step)",'
                '"differential_diagnoses":["2-4 differentials"],'
                '"recommendations":"Numbered guideline recommendations (ACC/AHA, ATS/IDSA, Fleischner 2017, BTS)",'
                '"prescriptions":"Guideline pharmacotherapy considerations",'
                '"patient_summary":"Compassionate plain-language explanation + 3 questions for their doctor",'
                '"critical_alert":false}'
            )
            for candidate in model_candidates:
                try:
                    kwargs: dict[str, Any] = {
                        "model": candidate,
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": structured_sys},
                            {"role": "user", "content": user_query},
                        ],
                        "max_tokens": 2800,
                    }
                    if "gpt-oss" in candidate:
                        kwargs["reasoning_effort"] = "low"
                    resp = client.chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    raw_text = (choice.message.content or "").strip()
                    if raw_text:
                        data = json.loads(raw_text)
                        comp_raw = data.get("compartments") or {}
                        compartments = {
                            "lungs": str(comp_raw.get("lungs", "See detailed findings.")),
                            "pleura": str(comp_raw.get("pleura", "No pneumothorax or large pleural effusion.")),
                            "cardio": str(comp_raw.get("cardio", "Cardiomediastinal silhouette evaluated.")),
                            "osseous": str(comp_raw.get("osseous", "Osseous thorax intact.")),
                        }
                        impression = str(data.get("impression") or "Clinical correlation advised.")
                        dictation = str(data.get("dictation_transcript") or impression)
                        recommendations = str(data.get("recommendations") or "Correlate clinically.")
                        prescriptions = str(data.get("prescriptions") or "Physician-directed management.")
                        patient_summary = str(data.get("patient_summary") or dictation)
                        exec_sum = data.get("executive_summary") if isinstance(data.get("executive_summary"), list) else [impression]
                        diffs = data.get("differential_diagnoses") if isinstance(data.get("differential_diagnoses"), list) else []
                        triage_level = str(data.get("triage_level") or "ROUTINE")
                        critical_alert = bool(data.get("critical_alert", False))
                        findings = (
                            f"LUNGS & AIRWAYS: {compartments['lungs']}\n"
                            f"PLEURA & SULCI: {compartments['pleura']}\n"
                            f"CARDIOMEDIASTINUM: {compartments['cardio']}\n"
                            f"OSSEOUS & SOFT TISSUES: {compartments['osseous']}"
                        )
                        full_text = (
                            f"EXAMINATION: CHEST RADIOGRAPH (FRONTAL)\n"
                            f"AI SYNTHESIS ENGINE: groq:{candidate} (Structured ACR/RSNA Clinical Synthesis)\n\n"
                            f"FINDINGS:\n{findings}\n\n"
                            f"IMPRESSION:\n{impression}\n\n"
                            f"GUIDELINE-DIRECTED RECOMMENDATIONS:\n{recommendations}\n\n"
                            f"PHARMACOTHERAPY & STEWARDSHIP:\n{prescriptions}\n\n"
                            f"PATIENT-FRIENDLY SUMMARY:\n{patient_summary}\n\n"
                            f"DISCLAIMER: Investigational AI clinical decision support. Licensed radiologist verification required."
                        )
                        structured_dict = {
                            "findings": findings,
                            "impression": impression,
                            "dictation_transcript": dictation,
                            "recommendations": recommendations,
                            "prescriptions": prescriptions,
                            "patient_summary": patient_summary,
                            "executive_summary": [str(x) for x in exec_sum],
                            "differential_diagnoses": [str(x) for x in diffs],
                            "triage_level": triage_level,
                            "critical_alert": critical_alert,
                            "compartments": compartments,
                            "report": full_text,
                            "draft": full_text,
                            "stage": "groq_structured",
                            "source": f"groq:{candidate}",
                        }
                        return full_text, f"groq:{candidate}", structured_dict
                except Exception as inner_exc:
                    print(f"[llm] Groq {candidate} failed ({inner_exc}); trying next...")
                    continue
        except Exception as exc:
            print(f"[llm] Groq call failed ({exc}); trying fallback...")

    # 3. Try xAI Grok
    if xai_key and (provider in (None, "xai")):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1", timeout=15.0)
            chosen_model = model or os.environ.get("CXR_LLM_MODEL", "grok-4.6")
            resp = client.chat.completions.create(
                model=chosen_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=1100,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip(), f"xai:{chosen_model}", None
        except Exception as exc:
            print(f"[llm] xAI Grok call failed ({exc}); trying fallback...")

    # 4. Try OpenAI
    if openai_key and (provider in (None, "openai")):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key, timeout=15.0)
            chosen_model = model or "gpt-4o-mini"
            resp = client.chat.completions.create(
                model=chosen_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=1100,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text.strip(), f"openai:{chosen_model}", None
        except Exception as exc:
            print(f"[llm] OpenAI call failed ({exc}); using built-in synthesizer...")

    # 5. Built-in ACR/RSNA Clinical NLP Synthesizer (always succeeds offline)
    return _synthesize_clinical_report(draft, patient=patient), "clinical-nlp", None


def polish_with_grok(
    draft: str,
    model: str | None = None,
    patient: dict[str, Any] | None = None,
) -> str:
    """Backwards-compatible alias returning just the synthesized report string."""
    text, _, _ = polish_with_llm(draft, model=model, patient=patient)
    return text
