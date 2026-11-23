"""Pydantic v2 response and request schemas for the CXR Diagnostic API v4.0."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    """Clinical context for personalized report synthesis."""

    age: int | None = Field(default=None, description="Patient age in years")
    sex: str | None = Field(default="Unspecified", description="Biological sex (Male / Female / Other)")
    setting: str | None = Field(default="Ambulatory / Outpatient", description="Clinical care setting (ED, ICU, Inpatient, Outpatient)")
    indication: str | None = Field(default="Thoracic diagnostic evaluation", description="Chief complaint / indication")
    comorbidities: str | None = Field(default="", description="Medical history / comorbidities")
    smoking: str | None = Field(default="", description="Tobacco / smoking history")


class PathologyPrediction(BaseModel):
    """Single pathology prediction with ensemble uncertainty & provenance."""

    name: str = Field(..., description="Pathology name, e.g. 'Cardiomegaly'")
    score: float = Field(..., ge=0.0, le=1.0, description="Calibrated fused probability")
    severity: str = Field(..., description="mild / moderate / severe")
    uncertainty: float = Field(default=0.02, description="Epistemic inter-model std dev")
    agreement: float = Field(default=0.95, description="Inter-backbone consensus score [0..1]")
    ci_low: float | None = Field(default=None, description="Lower 95% confidence bound")
    ci_high: float | None = Field(default=None, description="Upper 95% confidence bound")
    cnn_score: float | None = Field(default=None, description="Local multi-backbone CNN ensemble score")
    gemini_decision_score: float | None = Field(default=None, description="Gemini Decision 1 multimodal score")
    backbones: dict[str, float] = Field(default_factory=dict, description="Per-backbone calibrated probabilities")
    anatomical_zone: str | None = Field(default=None, description="Grad-CAM++ localized anatomical region")


class HeatmapData(BaseModel):
    """Grad-CAM++ localization artifact with hotspot coordinates & anatomical zone."""

    pathology: str
    image_base64: str = Field(..., description="Blended overlay PNG as base64")
    raw_heatmap_base64: str | None = Field(default=None, description="Pure transparent RGBA heatmap PNG")
    method: str = Field(default="Grad-CAM++", description="Attribution algorithm")
    colormap: str = Field(default="turbo", description="Colormap applied")
    peak_x: float | None = Field(default=None, description="Normalized hotspot peak X [0..1]")
    peak_y: float | None = Field(default=None, description="Normalized hotspot peak Y [0..1]")
    peak_intensity: float | None = Field(default=None, description="Peak activation intensity")
    bbox: list[float] | None = Field(default=None, description="Normalized hotspot bbox [x1, y1, x2, y2]")
    coverage_pct: float | None = Field(default=None, description="Activated spatial coverage %")
    anatomical_zone: str | None = Field(default=None, description="Localized thoracic zone")
    laterality: str | None = Field(default=None, description="Right / Left / Bilateral / Midline")


class AnatomyData(BaseModel):
    """Quantitative biometry from PSPNet thoracic organ segmentation."""

    ctr: float | None = Field(default=None, description="Cardiothoracic Ratio (cardiac width / thoracic width)")
    ctr_status: str = Field(default="Unavailable", description="Clinical interpretation of CTR")
    ctr_coords: dict[str, float] | None = Field(default=None, description="Normalized CTR caliper coordinates")
    left_lung_pct: float | None = Field(default=None)
    right_lung_pct: float | None = Field(default=None)
    lung_symmetry_ratio: float | None = Field(default=None)
    overlay_base64: str | None = Field(default=None, description="Transparent RGBA organ contours + CTR line")
    segmentation_ms: float = Field(default=0.0)
    organs_detected: list[str] = Field(default_factory=list)


class DecisionSummary(BaseModel):
    """Gemini Decision 1 closed-set multimodal judgment summary."""

    active: bool = False
    status: str = "inactive"
    model: str = "gemini-decision-1"
    transport: str | None = None
    latency_ms: float = 0.0
    output_tokens: int = 0
    is_valid_cxr: bool | None = None
    valid_cxr_probability: float | None = None
    triage_acuity: str | None = None
    triage_probabilities: dict[str, Any] = Field(default_factory=dict)
    projection_view: str | None = None
    image_quality_score: float | None = None
    message: str | None = None


class ReportData(BaseModel):
    """Generated ACR/RSNA radiology report."""

    findings: str = Field(default="", description="FINDINGS section text")
    impression: str = Field(default="", description="IMPRESSION section text")
    dictation_transcript: str = Field(default="", description="Natural spoken-style attending radiologist dictation transcript")
    recommendations: str = Field(default="", description="Guideline-directed clinical recommendations")
    prescriptions: str = Field(default="", description="Guideline-directed medication and prescription guidance")
    patient_summary: str = Field(default="", description="Plain-language patient friendly explanation")
    executive_summary: list[str] = Field(default_factory=list, description="Concise ED/ICU handoff bullets")
    differential_diagnoses: list[str] = Field(default_factory=list, description="Prioritized differentials")
    triage_level: str = Field(default="ROUTINE", description="STAT_CRITICAL / URGENT / SEMI_URGENT / ROUTINE")
    critical_alert: bool = Field(default=False, description="Emergency finding flag (pneumothorax, etc.)")
    compartments: dict[str, str] = Field(default_factory=dict, description="Anatomical compartments")
    full_text: str = Field(default="", description="Complete report draft")
    source: str = Field(default="template", description="Generation engine tag")


class PredictionResponse(BaseModel):
    """Full response for a single chest radiograph diagnostic study."""

    study_id: str | None = Field(default=None, description="Persisted SQLite study record ID if saved for a user")
    user_id: str | None = Field(default=None, description="Owner user/patient record ID")
    predictions: list[PathologyPrediction] = Field(default_factory=list, description="All 14 pathology scores")
    heatmaps: list[HeatmapData] = Field(default_factory=list, description="Grad-CAM++ overlays for positive findings")
    anatomy: AnatomyData | None = Field(default=None, description="PSPNet quantitative CTR & lung segmentation")
    decision_summary: DecisionSummary | None = Field(default=None, description="Gemini Decision 1 multimodal judgments")
    ensemble_info: dict[str, Any] = Field(default_factory=dict, description="Active backbones and execution flags")
    timings: dict[str, float] = Field(default_factory=dict, description="Per-stage latency telemetry (ms)")
    dicom_metadata: dict[str, Any] | None = Field(default=None, description="Extracted DICOM header tags if .dcm")
    report: ReportData
    patient: PatientProfile | None = Field(default=None, description="Patient clinical profile used during synthesis")
    inference_time_ms: float = Field(..., description="End-to-end latency in ms")
    vision_latency_ms: float | None = Field(default=None, description="Pure GPU vision + Grad-CAM++ latency in ms")


class BatchStudyItem(BaseModel):
    filename: str
    triage_level: str
    critical_alert: bool
    top_findings: list[PathologyPrediction]
    ctr: float | None = None
    vision_latency_ms: float
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    studies: list[BatchStudyItem]
    total_studies: int
    total_latency_ms: float
    avg_latency_ms: float
    stat_count: int


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    device: str = "cpu"
    gpu_name: str | None = None
    model_loaded: bool = False
    active_backbones: list[str] = Field(default_factory=list)
    segmentation_ready: bool = False
    gemini_configured: bool = False
    gemini_flash_model: str = "gemini-4-hydrogen"
    gemini_decision_enabled: bool = True
    batcher_active: bool = False
    uptime_seconds: float = 0.0


class SampleCaseResponse(BaseModel):
    """Metadata for verified clinical benchmark sample."""

    id: str
    name: str
    subtitle: str
    filename: str
    primary_findings: list[str] = Field(default_factory=list)
    suggested_patient: PatientProfile | None = Field(default=None)


class SingleHeatmapResponse(BaseModel):
    """Response for on-demand single pathology Grad-CAM++ heatmap."""

    pathology: str
    image_base64: str
    raw_heatmap_base64: str | None = None
    method: str = "Grad-CAM++"
    colormap: str = "turbo"
    peak_x: float | None = None
    peak_y: float | None = None
    bbox: list[float] | None = None
    anatomical_zone: str | None = None
    laterality: str | None = None


class CopilotChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    study_context: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, str]] = Field(default_factory=list)
    previous_interaction_id: str | None = None
    user_id: str | None = None
    study_id: str | None = None


class CopilotChatResponse(BaseModel):
    answer: str
    provider: str
    interaction_id: str | None = None
    latency_ms: float


class UserCreateRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    mrn: str | None = Field(default=None, max_length=40)
    email: str | None = Field(default="", max_length=160)
    role: str = Field(default="Patient", max_length=40)
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str = Field(default="Unspecified")
    setting: str = Field(default="Ambulatory / Outpatient")
    indication: str = Field(default="Thoracic diagnostic evaluation")
    comorbidities: str = Field(default="")
    smoking: str = Field(default="")
    notes: str = Field(default="")


class UserUpdateRequest(BaseModel):
    full_name: str | None = None
    mrn: str | None = None
    email: str | None = None
    role: str | None = None
    age: int | None = None
    sex: str | None = None
    setting: str | None = None
    indication: str | None = None
    comorbidities: str | None = None
    smoking: str | None = None
    preferred_theme: str | None = None
    preferred_threshold: float | None = Field(default=None, ge=0.1, le=0.95)
    preferred_colormap: str | None = None
    notes: str | None = None


class StudyUpdateRequest(BaseModel):
    study_name: str | None = None
    triage_level: str | None = None
    dictation_transcript: str | None = None
    physician_name: str | None = None
    physician_creds: str | None = None
    physician_notes: str | None = None
    signed_at: str | None = None
    report: dict[str, Any] | None = None


class ReportSynthesizeRequest(BaseModel):
    scores: dict[str, float]
    threshold: float = 0.50
    patient: PatientProfile | None = None
    anatomy: dict[str, Any] | None = None
    hotspots: list[dict[str, Any]] | None = None
    use_llm: bool = True


class AIConfigUpdate(BaseModel):
    gemini_api_key: str | None = None
    gemini_flash_model: str | None = None
    gemini_decision_enabled: bool | None = None
    gemini_flash_enabled: bool | None = None
    decision_fusion_weight: float | None = Field(default=None, ge=0.05, le=0.65)
    groq_api_key: str | None = None
    groq_model: str | None = None
    ensemble_mode: str | None = None
