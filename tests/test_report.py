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
