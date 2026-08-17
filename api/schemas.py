"""Pydantic v2 response and request schemas for the CXR Diagnostic API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PathologyPrediction(BaseModel):
    """Single pathology prediction with confidence and severity."""

    name: str = Field(..., description="Pathology name, e.g. 'Cardiomegaly'")
    score: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability")
    severity: str = Field(..., description="mild / moderate / severe")


class HeatmapData(BaseModel):
    """Base64-encoded Grad-CAM overlay for one pathology."""

    pathology: str
    image_base64: str = Field(..., description="PNG image as base64 string")


class ReportData(BaseModel):
    """Generated radiology report."""

    findings: str = Field(default="", description="FINDINGS section text")
    impression: str = Field(default="", description="IMPRESSION section text")
    full_text: str = Field(default="", description="Complete report draft")
    source: str = Field(
        default="template",
        description="Generation stage: template / t5 / llm",
    )


class PredictionResponse(BaseModel):
    """Full response for a single image prediction."""

    predictions: list[PathologyPrediction] = Field(
        default_factory=list,
        description="All 14 pathology scores",
    )
    heatmaps: list[HeatmapData] = Field(
        default_factory=list,
        description="Grad-CAM overlays for positive findings",
    )
    report: ReportData
    inference_time_ms: float = Field(..., description="End-to-end latency in ms")


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    device: str = "cpu"
    model_loaded: bool = False
    uptime_seconds: float = 0.0
