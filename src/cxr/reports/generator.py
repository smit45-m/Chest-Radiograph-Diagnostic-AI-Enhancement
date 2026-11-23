"""Multi-stage clinical report pipeline:
Stage 1 (ACR Template + Quantitative CTR + Grad-CAM++ Zones)
→ Optional Stage 2 (Local Hugging Face T5)
→ Stage 3 (Gemini Flash Multimodal / Groq / xAI / Built-in Clinical NLP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cxr.reports.composer import compose_report


def _build_executive_summary(
    report: dict[str, Any],
    anatomy: dict[str, Any] | None = None,
    hotspots: list[dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
) -> list[str]:
    bullets: list[str] = []
    positives = report.get("positives") or []
    if not positives:
        bullets.append("No acute cardiopulmonary pathology detected above operating threshold.")
    else:
        top_desc = ", ".join(
            f"{p['pathology'].replace('_', ' ')} ({p['score']*100:.0f}%)"
            for p in positives[:3]
        )
        bullets.append(f"Primary AI Findings: {top_desc}.")
        if hotspots:
            z = hotspots[0]
            bullets.append(
                f"Peak Grad-CAM++ Localization: {z.get('pathology', '').replace('_', ' ')} "
                f"in {z.get('anatomical_zone', 'thoracic parenchyma')} ({z.get('laterality', 'frontal')})."
            )
    if anatomy and anatomy.get("ctr") is not None:
        bullets.append(
            f"Automated PSPNet Biometry: Cardiothoracic Ratio (CTR) = {anatomy['ctr']:.2f} "
            f"({anatomy.get('ctr_status', 'measured')}), L/R lung symmetry {anatomy.get('lung_symmetry_ratio', 1.0):.2f}."
        )
    if decision_summary and decision_summary.get("active"):
        bullets.append(
            f"Gemini Decision 1 Triage: {decision_summary.get('triage_acuity', 'ROUTINE')} | "
            f"View: {decision_summary.get('projection_view', 'PA')} | "
            f"Quality: {decision_summary.get('image_quality_score', 4.0)}/5."
        )
    return bullets[:4]


def _derive_triage_level(
    positives: list[dict[str, Any]],
    decision_summary: dict[str, Any] | None = None,
) -> str:
    if decision_summary and decision_summary.get("active") and decision_summary.get("triage_acuity"):
        return str(decision_summary["triage_acuity"])
    names = {p["pathology"] for p in positives}
    if "Pneumothorax" in names or any(p["pathology"] == "Edema" and p["score"] >= 0.72 for p in positives):
        return "STAT_CRITICAL"
    if names.intersection({"Pneumonia", "Consolidation", "Mass", "Effusion"}):
        return "URGENT"
    if positives:
        return "SEMI_URGENT"
    return "ROUTINE"


def generate_report(
    scores: dict[str, float],
    threshold: float = 0.45,
    image_id: str | None = None,
    use_t5: bool = True,
    use_llm: bool = True,
    t5_dir: str | Path = "checkpoints/report_t5",
    patient: dict[str, Any] | None = None,
    image_bytes: bytes | None = None,
    anatomy: dict[str, Any] | None = None,
    hotspots: list[dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = compose_report(scores, threshold=threshold, image_id=image_id, patient=patient)

    # Enrich cardiac compartment with quantitative CTR if measured
    if anatomy and anatomy.get("ctr") is not None:
        ctr_note = f" Quantitative cardiothoracic ratio (CTR) measured via anatomical segmentation is {anatomy['ctr']:.2f} ({anatomy.get('ctr_status', '')})."
        report["compartments"]["cardio"] = report["compartments"].get("cardio", "") + ctr_note
        report["findings"] = (
            f"LUNGS & AIRWAYS: {report['compartments']['lungs']}\n"
            f"PLEURA & SULCI: {report['compartments']['pleura']}\n"
            f"CARDIOMEDIASTINUM: {report['compartments']['cardio']}\n"
            f"OSSEOUS & SOFT TISSUES: {report['compartments']['osseous']}"
        )

    # Enrich lung/pleura with Grad-CAM++ anatomical zone notes
    if hotspots:
        zone_notes = [
            f"{h['pathology'].replace('_', ' ')} localizes to {h['anatomical_zone']}"
            for h in hotspots[:3]
            if h.get("anatomical_zone")
        ]
        if zone_notes:
            report["findings"] += "\nSPATIAL LOCALIZATION (Grad-CAM++): " + "; ".join(zone_notes) + "."

    text = report["draft"]
    stage = "template"
    source = "template"

    if use_t5:
        try:
            from cxr.reports.transformer import rewrite_with_t5

            text = rewrite_with_t5(text, ckpt_dir=t5_dir)
            stage = "t5"
            source = "t5"
        except Exception:
            pass

    if use_llm:
        try:
            from cxr.reports.llm import polish_with_llm

            polished, provider_tag, gemini_structured = polish_with_llm(
                text if stage != "template" else report["draft"],
                patient=patient,
                scores=scores,
                threshold=threshold,
                image_bytes=image_bytes,
                anatomy=anatomy,
                hotspots=hotspots,
                decision_summary=decision_summary,
            )
            if gemini_structured:
                # Gemini Flash returned rich structured ACR JSON — merge cleanly
                report.update(gemini_structured)
                return report

            text = polished
            stage = "llm"
            source = provider_tag
        except Exception:
            pass

    report["report"] = text
    report["stage"] = stage
    report["source"] = source
    report["executive_summary"] = _build_executive_summary(
        report, anatomy=anatomy, hotspots=hotspots, decision_summary=decision_summary
    )
    report["triage_level"] = _derive_triage_level(report.get("positives", []), decision_summary)
    report["differential_diagnoses"] = [
        p["pathology"].replace("_", " ") for p in report.get("positives", [])[:4]
    ] or ["Normal frontal chest radiograph"]
    ctr_clause = (
        f" Automated anatomical segmentation measures a cardiothoracic ratio of {anatomy['ctr']:.2f}"
        f" ({anatomy.get('ctr_status', 'measured')})."
        if anatomy and anatomy.get("ctr") is not None
        else ""
    )
    zone_clause = (
        f" Peak Grad-CAM++ activation localizes to the {hotspots[0]['anatomical_zone']}."
        if hotspots and hotspots[0].get("anatomical_zone")
        else ""
    )
    patient_lead = (
        f"Frontal chest radiograph for this {patient.get('age')}-year-old {str(patient.get('sex', '')).lower()} patient "
        f"in {patient.get('setting', 'clinical care')}"
        if patient and patient.get("age")
        else "Frontal chest radiograph"
    )
    positives_list = report.get("positives") or []
    if positives_list:
        pos_desc = ", ".join(
            f"{p['pathology'].replace('_', ' ').lower()} ({p['score']*100:.0f}% confidence)"
            for p in positives_list[:3]
        )
        finding_sentence = f" demonstrates {pos_desc}."
        action_sentence = " Prompt clinical correlation and guideline-directed management are recommended."
    else:
        finding_sentence = " demonstrates clear lung fields without focal consolidation, pleural effusion, or acute cardiopulmonary abnormality."
        action_sentence = " Routine clinical follow-up is advised as clinically indicated."

    report["dictation_transcript"] = (
        f"{patient_lead}{finding_sentence}{zone_clause}{ctr_clause}{action_sentence}"
    ).strip()
    return report
