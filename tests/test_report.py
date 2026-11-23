"""Tests for report generation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.constants import PATHOLOGIES
from cxr.reports.composer import compose_report
from cxr.reports.generator import generate_report


def test_compose_report_no_findings():
    """Zero scores should produce a 'no acute' report."""
    scores = {name: 0.0 for name in PATHOLOGIES}
    result = compose_report(scores, threshold=0.45)
    assert "No acute" in result["impression"] or "no acute" in result["impression"].lower()
    assert result["positives"] == []


def test_compose_report_with_findings():
    """High scores should include pathology names in the report."""
    scores = {name: 0.1 for name in PATHOLOGIES}
    scores["Cardiomegaly"] = 0.92
    scores["Effusion"] = 0.78
    result = compose_report(scores, threshold=0.45)
    assert len(result["positives"]) == 2
    names = [p["pathology"] for p in result["positives"]]
    assert "Cardiomegaly" in names
    assert "Effusion" in names


def test_compose_report_severity():
    """Score thresholds should map to correct severity bands."""
    scores = {name: 0.0 for name in PATHOLOGIES}
    scores["Pneumonia"] = 0.90  # severe
    scores["Effusion"] = 0.70  # moderate
    scores["Nodule"] = 0.50   # mild
    result = compose_report(scores, threshold=0.45)

    severity_map = {p["pathology"]: p["severity"] for p in result["positives"]}
    assert severity_map["Pneumonia"] == "severe"
    assert severity_map["Effusion"] == "moderate"
    assert severity_map["Nodule"] == "mild"


def test_generate_report_template_fallback():
    """With T5 and LLM disabled, source should be 'template'."""
    scores = {name: 0.0 for name in PATHOLOGIES}
    scores["Atelectasis"] = 0.60
    result = generate_report(scores, use_t5=False, use_llm=False)
    assert result["source"] == "template"
    assert "report" in result or "draft" in result


def test_report_structure():
    """Report dict should have all required keys."""
    scores = {name: 0.5 for name in PATHOLOGIES}
    result = compose_report(scores, threshold=0.45)
    required_keys = {"findings", "impression", "positives", "draft", "source"}
    assert required_keys.issubset(result.keys())


def test_compose_report_with_patient_context():
    """Report should dynamically tailor findings, guidelines, and patient summary."""
    scores = {name: 0.05 for name in PATHOLOGIES}
    scores["Pneumonia"] = 0.88
    scores["Consolidation"] = 0.72

    patient = {
        "age": 74,
        "sex": "Male",
        "setting": "Emergency Department (ED)",
        "indication": "High fever, productive cough, pleuritic chest pain",
        "comorbidities": "Type 2 Diabetes, Hypertension",
        "smoking": "Former smoker",
    }
    result = compose_report(scores, threshold=0.45, patient=patient)

    assert "74yo Male" in result["draft"]
    assert "Emergency Department" in result["draft"]
    assert "LUNGS & AIRWAYS:" in result["findings"]
    assert "CARDIOMEDIASTINUM:" in result["findings"]
    assert "compartments" in result
    assert "recommendations" in result
    assert "patient_summary" in result
    assert "ATS/IDSA" in result["recommendations"] or "antimicrobial" in result["recommendations"].lower()
    assert "What Your X-Ray Shows" in result["patient_summary"] or "What Was Detected" in result["patient_summary"]


def test_compose_report_critical_alert():
    """Pneumothorax should trigger critical alert flag."""
    scores = {name: 0.0 for name in PATHOLOGIES}
    scores["Pneumothorax"] = 0.92
    result = compose_report(scores, threshold=0.45)
    assert result["critical_alert"] is True
    assert "tension physiology" in result["draft"].lower()


def test_compose_report_prescriptions_change_by_condition():
    """Prescriptions must change depending on which pathologies are detected."""
    # 1. Pneumonia
    pneu_scores = {name: 0.0 for name in PATHOLOGIES}
    pneu_scores["Pneumonia"] = 0.85
    r_pneu = compose_report(pneu_scores, threshold=0.45, patient={"age": 70, "comorbidities": "Diabetes"})
    assert "ANTIMICROBIAL" in r_pneu["prescriptions"]
    assert "Amoxicillin" in r_pneu["prescriptions"] or "Augmentin" in r_pneu["prescriptions"]

    # 2. Cardiomegaly / Heart Failure
    cardio_scores = {name: 0.0 for name in PATHOLOGIES}
    cardio_scores["Cardiomegaly"] = 0.88
    r_cardio = compose_report(cardio_scores, threshold=0.45)
    assert "Furosemide" in r_cardio["prescriptions"] or "Lasix" in r_cardio["prescriptions"]
    assert "NSAIDs" in r_cardio["prescriptions"]  # contraindication alert

    # 3. Solitary Nodule
    nodule_scores = {name: 0.0 for name in PATHOLOGIES}
    nodule_scores["Nodule"] = 0.82
    r_nodule = compose_report(nodule_scores, threshold=0.45)
    assert "Do NOT prescribe empiric antibiotics" in r_nodule["prescriptions"]

    # 4. Normal / Unremarkable
    clear_scores = {name: 0.0 for name in PATHOLOGIES}
    r_clear = compose_report(clear_scores, threshold=0.45)
    assert "NO CARDIOPULMONARY MEDICATIONS" in r_clear["prescriptions"]


