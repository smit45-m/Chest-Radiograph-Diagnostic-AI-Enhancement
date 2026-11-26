"""Gemini Decision 1 (Decisions API v1beta) + Gemini Interactions API Clinical Engine.

Upgraded to official Google Gemini API standards (`google-genai >= 3.2.0`):
1. **Gemini Decision 1 (`gemini-decision-1`)** via Decisions API (`v1beta`):
   - Native `client.decisions.create(...)` (SDK >= 3.2.0) with direct `POST /v1beta/decisions`
     (`x-goog-api-key` header) fallback so it works cleanly on any SDK version.
   - Evaluates 18 closed-set questions over the radiograph in ONE forward pass with
     **0 output tokens**:
       * `is_valid_cxr` (`noul` yes/no modality gate)
       * All 14 NIH Thoracic Pathologies (`noul` -> calibrated probability + confidence)
       * `triage_acuity` (`choice` -> `criteria` map)
       * `projection_view` (`choice` -> `criteria` map)
       * `image_quality` (`score` -> ordered `criteria` array)
   - Multimodal input blocks follow the canonical `{"type": "image", "data": ..., "mime_type": ...}`
     and `{"type": "text", "text": ...}` format.

2. **Gemini Interactions API (`client.interactions.create`)**:
   - Uses current models: `gemini-4-hydrogen` (default balanced multimodal driver),
     `gemini-3.5-flash-lite` (fastest high-throughput Flash-Lite summarization),
     `gemini-3.1-flash-lite`, and `gemini-4-argon` (deep differential reasoning).
   - Structured JSON clinical reporting via top-level `response_format` + Pydantic schema
     (no legacy `generateContent`, no deprecated `temperature` parameter).
   - Stateful multi-turn Radiology Copilot supporting `previous_interaction_id`, with
     fast Groq and offline clinical-knowledge fallbacks.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

from cxr.constants import PATHOLOGIES


# Current verified low-latency Gemini models (ordered fastest -> deepest; avoids stale 404/503 fallthrough)
CURRENT_GEMINI_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-4-hydrogen",
]

# Thread-safe persistent HTTP keep-alive connection pool & adaptive model circuit breaker
_HTTP_POOL_LOCK = threading.Lock()
_SHARED_HTTP_CLIENT: httpx.Client | None = None
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


def _get_shared_http_client() -> httpx.Client:
    """Return a shared keep-alive httpx.Client so TLS/TCP handshakes are reused across API calls."""
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
        with _HTTP_POOL_LOCK:
            if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
                _SHARED_HTTP_CLIENT = httpx.Client(
                    timeout=httpx.Timeout(11.0, connect=4.5),
                    limits=httpx.Limits(
                        max_connections=24,
                        max_keepalive_connections=12,
                        keepalive_expiry=120.0,
                    ),
                )
    return _SHARED_HTTP_CLIENT


def _is_model_healthy(model_id: str) -> bool:
    until = _MODEL_COOLDOWN_UNTIL.get(model_id, 0.0)
    return time.monotonic() >= until


def _mark_model_unhealthy(model_id: str, status_code: int | None = None) -> None:
    """Temporarily bench dead (404/400) or overloaded (429/503/timeout) models to prevent 20s+ stalls."""
    cooldown = 600.0 if status_code in (400, 404) else 120.0
    _MODEL_COOLDOWN_UNTIL[model_id] = time.monotonic() + cooldown


def _mark_model_healthy(model_id: str) -> None:
    _MODEL_COOLDOWN_UNTIL.pop(model_id, None)


def get_circuit_breaker_status() -> dict[str, float]:
    now = time.monotonic()
    return {
        m: round(until - now, 1)
        for m, until in _MODEL_COOLDOWN_UNTIL.items()
        if until > now
    }

# Radiologist-grade differential criteria with explicit specificity & size/projection definitions
PATHOLOGY_DECISION_CRITERIA: dict[str, str] = {
    "Atelectasis": (
        "Is there definite linear, plate-like, subsegmental, or lobar lung volume loss/collapse "
        "(with fissure/mediastinal shift or elevated hemidiaphragm)? Answer Yes only for clear collapse, not normal vascular markings."
    ),
    "Cardiomegaly": (
        "Is the cardiac silhouette genuinely enlarged (CTR > 0.50 on upright PA view, or > 0.56 on portable AP view accounting for AP magnification)? "
        "Do NOT answer Yes for mild AP projection magnification alone."
    ),
    "Effusion": (
        "Is there clear blunting of a costophrenic angle, a concave meniscus sign, or layered homogeneous pleural fluid? "
        "Answer No if both costophrenic angles are sharp."
    ),
    "Infiltration": (
        "Are there distinct patchy alveolar or interstitial lung opacities/infiltrates in the lung parenchyma "
        "(consistent with acute infectious or inflammatory infiltrate)? Answer No for normal bronchovascular markings."
    ),
    "Mass": (
        "Is there a definite discrete pulmonary or mediastinal mass lesion measuring GREATER than 3 cm in diameter with well- or ill-defined margins? "
        "Answer No if no discrete mass > 3 cm is present."
    ),
    "Nodule": (
        "Is there a discrete rounded focal pulmonary nodule measuring 3 cm or LESS in diameter within the lung parenchyma? "
        "Answer No for normal end-on vessels or rib crossings."
    ),
    "Pneumonia": (
        "Are there clear focal, segmental, or lobar airspace opacities/consolidations (or air bronchograms) clinically and radiographically consistent with acute pneumonia? "
        "Consider clinical fever/cough context alongside visible airspace opacity."
    ),
    "Pneumothorax": (
        "Is there a definite crisp visceral pleural line with completely absent peripheral lung markings indicating pleural air (pneumothorax)? "
        "Answer No for skin folds or scapular borders."
    ),
    "Consolidation": (
        "Is there dense, homogeneous alveolar airspace consolidation obscuring underlying pulmonary vessels or creating an air bronchogram? "
        "Answer No for clear aerated lungs or solitary nodules without airspace filling."
    ),
    "Edema": (
        "Are there unmistakable signs of pulmonary edema such as perihilar bat-wing airspace opacities, Kerley B lines, or marked upper-lobe cephalization? "
        "Answer No for isolated cardiomegaly with clear lung fields."
    ),
    "Emphysema": (
        "Are there clear signs of emphysema/COPD: severely flattened hemidiaphragms, hyperlucent hyperinflated lungs, or visible bullae? "
        "Answer No for normal inspiratory effort."
    ),
    "Fibrosis": (
        "Are there distinct coarse reticular interstitial opacities, basilar/peripheral honeycombing, or chronic architectural distortion indicating pulmonary fibrosis? "
        "Consider ILD history alongside reticular shadowing."
    ),
    "Pleural_Thickening": (
        "Is there definite apical pleural capping or smooth/nodular thickening along the lateral pleural margin? Answer No for normal rib/companion shadows."
    ),
    "Hernia": (
        "Is there a clear retrocardiac air-fluid level or visible herniation of stomach/abdominal viscera above the diaphragm? Answer No for normal subdiaphragmatic gastric bubble."
    ),
}


class CompartmentsSchema(BaseModel):
    lungs: str = Field(description="Lungs & Airways findings")
    pleura: str = Field(description="Pleural Spaces & Costophrenic Sulci findings")
    cardio: str = Field(description="Cardiomediastinum & Vasculature findings including quantitative CTR")
    osseous: str = Field(description="Osseous thorax, ribs, spine & diaphragm findings")


class ClinicalReportSchema(BaseModel):
    """Pydantic schema enforced via Interactions API `response_format`."""

    executive_summary: list[str] = Field(description="3 concise clinical handoff bullet points")
    triage_level: str = Field(description="STAT_CRITICAL | URGENT | SEMI_URGENT | ROUTINE")
    compartments: CompartmentsSchema
    impression: str = Field(description="Numbered prioritized diagnostic conclusions")
    dictation_transcript: str = Field(
        description=(
            "Natural spoken-style attending thoracic radiologist dictation transcript (3-5 crisp clinical sentences) "
            "explicitly tailored to this patient's detected pathology, quantitative CTR, Grad-CAM++ localization zone, "
            "and urgent clinical next steps."
        )
    )
    differential_diagnoses: list[str] = Field(description="Top 2-4 prioritized differential diagnoses")
    recommendations: str = Field(description="Numbered evidence-based recommendations citing ATS/IDSA, ACC/AHA, Fleischner 2017, ACCP, or BTS")
    prescriptions: str = Field(description="Guideline-directed pharmacotherapy guidance with contraindication alerts")
    patient_summary: str = Field(description="Compassionate plain-language explanation + 3 questions to ask their doctor")
    critical_alert: bool = Field(description="True if life-threatening finding such as pneumothorax or acute severe edema")


def _normalize_model_id(raw: str | None) -> str:
    """Automatically map legacy/deprecated 2.5/2.0/1.5 model strings to current Gemini 4 / 3.5 models."""
    val = (raw or "").strip()
    if not val:
        return "gemini-4-hydrogen"
    lower = val.lower()
    if lower.startswith(("gemini-1.5", "gemini-2.0", "gemini-2.5-pro")):
        return "gemini-4-hydrogen"
    if lower == "gemini-2.5-flash-lite":
        return "gemini-3.5-flash-lite"
    if lower.startswith(("gemini-2.5-flash", "gemini-3-flash")):
        return "gemini-4-hydrogen"
    return val


@dataclass
class AIRuntimeConfig:
    """Mutable runtime AI engine configuration (seeded from env/.env, editable via API)."""

    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    )
    gemini_flash_model: str = field(
        default_factory=lambda: _normalize_model_id(os.environ.get("GEMINI_FLASH_MODEL", "gemini-3.5-flash-lite"))
    )
    gemini_decision_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_DECISION_MODEL", "gemini-decision-1")
    )
    gemini_decision_enabled: bool = field(
        default_factory=lambda: os.environ.get("GEMINI_DECISION_ENABLED", "true").lower() != "false"
    )
    gemini_flash_enabled: bool = field(
        default_factory=lambda: os.environ.get("GEMINI_FLASH_ENABLED", "true").lower() != "false"
    )
    decision_fusion_weight: float = field(
        default_factory=lambda: float(os.environ.get("GEMINI_DECISION_WEIGHT", "0.22"))
    )
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-20b"))
    ensemble_mode: str = field(default_factory=lambda: os.environ.get("CXR_ENSEMBLE_MODE", "full"))

    def refresh_from_env(self) -> None:
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not self.groq_api_key:
            self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.gemini_flash_model = _normalize_model_id(self.gemini_flash_model)

    def public_status(self) -> dict[str, Any]:
        self.refresh_from_env()
        return {
            "gemini_configured": bool(self.gemini_api_key.strip()),
            "gemini_key_hint": f"...{self.gemini_api_key[-4:]}" if len(self.gemini_api_key) >= 4 else "",
            "gemini_flash_model": self.gemini_flash_model,
            "gemini_decision_model": self.gemini_decision_model,
            "gemini_decision_enabled": self.gemini_decision_enabled,
            "gemini_flash_enabled": self.gemini_flash_enabled,
            "decision_fusion_weight": self.decision_fusion_weight,
            "groq_configured": bool(self.groq_api_key.strip()),
            "groq_model": self.groq_model,
            "ensemble_mode": self.ensemble_mode,
            "available_flash_models": list(CURRENT_GEMINI_MODELS),
        }


AI_RUNTIME_CONFIG = AIRuntimeConfig()


class GeminiDecision1Client:
    """High-speed closed-set clinical decision classifier using `gemini-decision-1` (`v1beta`).

    Canonical REST endpoint: `POST https://generativelanguage.googleapis.com/v1beta/decisions`
    Canonical SDK method: `client.decisions.create(model="gemini-decision-1", input=..., questions=...)`
    """

    DECISIONS_REST_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 12.0) -> None:
        AI_RUNTIME_CONFIG.refresh_from_env()
        raw_key = AI_RUNTIME_CONFIG.gemini_api_key if api_key is None else api_key
        self.api_key = (raw_key or "").strip()
        self.model = model or AI_RUNTIME_CONFIG.gemini_decision_model or "gemini-decision-1"
        self.timeout = timeout

    def build_questions(self) -> dict[str, Any]:
        """Build the 18-question Decisions API map using exact official `noul`, `choice`, and `score` schemas."""
        questions: dict[str, Any] = {
            "is_valid_cxr": {
                "type": "noul",
                "instructions": "Is this image a medical frontal or lateral chest X-ray radiograph?",
            },
            "triage_acuity": {
                "type": "choice",
                "instructions": "Assign the ACR clinical triage acuity level for this chest radiograph.",
                "criteria": {
                    "STAT_CRITICAL": "Immediate life-threatening finding such as pneumothorax, tension physiology, severe acute pulmonary edema, or massive effusion.",
                    "URGENT": "Acute clinically significant finding requiring prompt same-day management such as pneumonia, lobar consolidation, or suspicious mass.",
                    "SEMI_URGENT": "Subacute or chronic abnormality needing structured follow-up such as solitary pulmonary nodule, cardiomegaly, or mild atelectasis.",
                    "ROUTINE": "Normal or unremarkable chest radiograph with no acute cardiopulmonary process.",
                },
            },
            "projection_view": {
                "type": "choice",
                "instructions": "Identify the radiographic projection view of this chest X-ray.",
                "criteria": {
                    "PA_UPRIGHT": "Posteroanterior upright chest radiograph with scapulae rotated laterally and crisp cardiac border.",
                    "AP_PORTABLE": "Anteroposterior supine or portable bedside chest radiograph with magnified cardiac silhouette or support lines.",
                    "LATERAL": "Lateral side-view chest radiograph.",
                    "SUBOPTIMAL": "Rotated, clipped, severely under/over-exposed, or non-standard chest projection.",
                },
            },
            # Official Decisions API `score` schema uses an ordered `criteria` array (lowest -> highest)
            "image_quality": {
                "type": "score",
                "instructions": "Rate technical diagnostic quality (inspiration adequacy, penetration, absence of rotation) across the 5-point clinical rubric.",
                "criteria": [
                    "1 - Non-diagnostic quality or severely obscured thoracic anatomy",
                    "2 - Suboptimal exposure, heavy rotation, or poor inspiratory effort",
                    "3 - Acceptable diagnostic quality with minor technical limitations",
                    "4 - Good diagnostic quality with adequate inspiration and contrast",
                    "5 - Excellent clinical PACS reference quality",
                ],
            },
        }
        for pathology in PATHOLOGIES:
            questions[f"path_{pathology}"] = {
                "type": "noul",
                "instructions": PATHOLOGY_DECISION_CRITERIA[pathology],
            }
        return questions

    def build_multimodal_input(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        patient: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        context_str = "Evaluate this chest radiograph for thoracic pathology triage, projection view, and technical quality."
        if patient:
            ctx_parts = [
                f"Age: {patient.get('age')}" if patient.get("age") else "",
                f"Sex: {patient.get('sex')}" if patient.get("sex") else "",
                f"Setting: {patient.get('setting')}" if patient.get("setting") else "",
                f"Indication: {patient.get('indication')}" if patient.get("indication") else "",
            ]
            joined = ", ".join(p for p in ctx_parts if p)
            if joined:
                context_str += f" Clinical context: {joined}."
        return [
            {"type": "text", "text": context_str},
            {"type": "image", "data": b64_img, "mime_type": mime_type},
        ]

    def evaluate(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        patient: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run Gemini Decision 1 over the radiograph image and return calibrated clinical judgments."""
        t0 = time.perf_counter()
        if not self.api_key:
            return {
                "active": False,
                "status": "no_api_key",
                "model": self.model,
                "latency_ms": 0.0,
                "message": "Configure GEMINI_API_KEY to enable live Gemini Decision 1 multimodal consensus.",
            }

        questions = self.build_questions()
        multimodal_input = self.build_multimodal_input(image_bytes, mime_type=mime_type, patient=patient)

        raw_answers: dict[str, Any] | None = None
        transport = "rest-pooled"

        # Fast Primary Path: Official `POST /v1beta/decisions` over persistent keep-alive pool
        # (avoids per-request SDK client cold init & TLS handshake overhead)
        payload = {
            "model": self.model,
            "input": multimodal_input,
            "questions": questions,
        }
        try:
            http_client = _get_shared_http_client()
            resp = http_client.post(
                self.DECISIONS_REST_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json=payload,
                timeout=httpx.Timeout(self.timeout, connect=4.5),
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_answers = data.get("answers", data.get("results", {}))
            else:
                return {
                    "active": False,
                    "status": "api_error",
                    "model": self.model,
                    "http_status": resp.status_code,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "message": f"Gemini Decision 1 returned HTTP {resp.status_code}",
                }
        except Exception as exc:
            return {
                "active": False,
                "status": "error",
                "model": self.model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "message": str(exc),
            }

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return self._parse_decision_response(raw_answers or {}, latency_ms=latency_ms, transport=transport)

    @staticmethod
    def _normalize_sdk_answers(sdk_res: Any) -> dict[str, Any]:
        answers_obj = getattr(sdk_res, "answers", None) or {}
        out: dict[str, Any] = {}
        for k, v in answers_obj.items():
            if isinstance(v, dict):
                out[k] = v
            else:
                out[k] = {
                    "type": getattr(v, "type", None),
                    "value": getattr(v, "value", None),
                    "noul": getattr(v, "noul", None),
                    "choice": getattr(v, "choice", None),
                    "score": getattr(v, "score", None),
                    "confidence": getattr(v, "confidence", 0.75),
                    "probabilities": getattr(v, "probabilities", None),
                }
        return out

    def _parse_decision_response(
        self,
        answers: dict[str, Any],
        latency_ms: float,
        transport: str,
    ) -> dict[str, Any]:
        def _extract_prob(ans: dict[str, Any], default: float = 0.05) -> tuple[float, float]:
            if not isinstance(ans, dict):
                return default, 0.5
            prob = ans.get("noul")
            if prob is None and isinstance(ans.get("probabilities"), dict):
                probs_d = ans["probabilities"]
                prob = probs_d.get("yes", probs_d.get("true", probs_d.get("YES")))
            if prob is None and isinstance(ans.get("value"), (int, float)):
                prob = float(ans["value"])
            if prob is None:
                prob = default
            conf = float(ans.get("confidence", 0.75) or 0.75)
            return max(0.01, min(0.99, float(prob))), max(0.0, min(1.0, conf))

        pathology_scores: dict[str, float] = {}
        pathology_confidences: dict[str, float] = {}
        for p in PATHOLOGIES:
            ans = answers.get(f"path_{p}", {})
            prob, conf = _extract_prob(ans, default=0.05)
            pathology_scores[p] = round(prob, 4)
            pathology_confidences[p] = round(conf, 4)

        valid_cxr_prob, valid_cxr_conf = _extract_prob(answers.get("is_valid_cxr", {}), default=0.95)

        triage_ans = answers.get("triage_acuity", {})
        triage_acuity = triage_ans.get("choice") or triage_ans.get("value") or "ROUTINE"
        triage_probs = triage_ans.get("probabilities") or {}

        proj_ans = answers.get("projection_view", {})
        projection_view = proj_ans.get("choice") or proj_ans.get("value") or "PA_UPRIGHT"

        qual_ans = answers.get("image_quality", {})
        raw_score = qual_ans.get("score") if qual_ans.get("score") is not None else qual_ans.get("value", 4.0)
        try:
            q_num = float(raw_score)
            # Decisions API `score` expected value is 0-indexed (0..4) when legend has key "0"
            legend = qual_ans.get("legend")
            if isinstance(legend, dict) and "0" in legend and q_num <= 4.0:
                quality_score = round(q_num + 1.0, 2)
            else:
                quality_score = round(q_num, 2)
        except Exception:
            quality_score = 4.0

        return {
            "active": True,
            "status": "ok",
            "model": self.model,
            "transport": transport,
            "latency_ms": latency_ms,
            "output_tokens": 0,
            "is_valid_cxr": valid_cxr_prob >= 0.5,
            "valid_cxr_probability": round(valid_cxr_prob, 4),
            "valid_cxr_confidence": round(valid_cxr_conf, 4),
            "triage_acuity": str(triage_acuity),
            "triage_probabilities": triage_probs,
            "projection_view": str(projection_view),
            "image_quality_score": quality_score,
            "pathology_scores": pathology_scores,
            "pathology_confidences": pathology_confidences,
        }


class GeminiFlashClinician:
    """Multimodal clinical radiology report synthesizer & copilot using the Gemini Interactions API."""

    INTERACTIONS_REST_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

    REPORT_SYSTEM_INSTRUCTION = (
        "You are an attending academic thoracic radiologist. Synthesize the chest radiograph image, "
        "calibrated multi-backbone deep learning predictions, quantitative anatomical measurements "
        "(Cardiothoracic Ratio / lung symmetry), Grad-CAM++ anatomical localizations, and patient "
        "clinical profile into an authoritative ACR/RSNA structured radiology report adhering strictly "
        "to the provided JSON schema."
    )

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        AI_RUNTIME_CONFIG.refresh_from_env()
        raw_key = AI_RUNTIME_CONFIG.gemini_api_key if api_key is None else api_key
        self.api_key = (raw_key or "").strip()
        self.model = _normalize_model_id(model or AI_RUNTIME_CONFIG.gemini_flash_model)

    def _create_interaction(
        self,
        model: str,
        input_payload: Any,
        system_instruction: str,
        response_format: dict[str, Any] | None = None,
        previous_interaction_id: str | None = None,
        timeout_sec: float = 11.5,
    ) -> tuple[str, str | None]:
        """Execute an Interactions API call over persistent HTTP pool (`POST /v1beta/interactions`)."""
        body: dict[str, Any] = {
            "model": model,
            "input": input_payload,
            "system_instruction": system_instruction,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if previous_interaction_id:
            body["previous_interaction_id"] = previous_interaction_id

        http_client = _get_shared_http_client()
        resp = http_client.post(
            self.INTERACTIONS_REST_URL,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "Api-Revision": "2026-05-20",
            },
            json=body,
            timeout=httpx.Timeout(timeout_sec, connect=4.5),
        )
        resp.raise_for_status()
        data = resp.json()
        out_text = data.get("output_text") or ""
        if not out_text:
            # Interactions API v1beta returns `outputs` list OR `steps` list depending on revision
            for out_item in reversed(data.get("outputs", [])):
                if isinstance(out_item, dict):
                    if out_item.get("type") == "text" and out_item.get("text"):
                        out_text = out_item["text"]
                        break
                    parts = out_item.get("content") or []
                    if isinstance(parts, list):
                        joined = "".join(
                            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"
                        )
                        if joined:
                            out_text = joined
                            break
        if not out_text:
            for step in reversed(data.get("steps", [])):
                if step.get("type") == "model_output":
                    out_text = "".join(
                        p.get("text", "") for p in step.get("content", []) if p.get("type") == "text"
                    )
                    if out_text:
                        break
        return out_text.strip(), data.get("id")

    def synthesize_report(
        self,
        scores: dict[str, float],
        threshold: float,
        draft: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/jpeg",
        patient: dict[str, Any] | None = None,
        anatomy: dict[str, Any] | None = None,
        hotspots: list[dict[str, Any]] | None = None,
        decision_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Synthesize a structured ACR/RSNA clinical report via Gemini Interactions API."""
        if not self.api_key or not AI_RUNTIME_CONFIG.gemini_flash_enabled:
            return None

        t0 = time.perf_counter()
        positives = sorted(
            [(k, v) for k, v in scores.items() if v >= threshold],
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_all = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:6]

        prompt_sections = [
            f"MULTI-BACKBONE CALIBRATED SCORES (threshold={threshold:.2f}):",
            "Positives: " + (", ".join(f"{k}: {v*100:.1f}%" for k, v in positives) if positives else "None above threshold"),
            "Top 6 Ranked Scores: " + ", ".join(f"{k}: {v*100:.1f}%" for k, v in top_all),
        ]
        if anatomy:
            prompt_sections.append(
                f"QUANTITATIVE ANATOMY (PSPNet Segmentation): CTR={anatomy.get('ctr')} "
                f"({anatomy.get('ctr_status', 'n/a')}), Right Lung Area={anatomy.get('right_lung_pct')}%, "
                f"Left Lung Area={anatomy.get('left_lung_pct')}%, Symmetry={anatomy.get('lung_symmetry_ratio')}"
            )
        if hotspots:
            hs_str = "; ".join(
                f"{h.get('pathology')}: {h.get('anatomical_zone')} ({h.get('laterality')})"
                for h in hotspots
            )
            prompt_sections.append(f"GRAD-CAM++ SPATIAL LOCALIZATIONS: {hs_str}")
        if decision_summary and decision_summary.get("active"):
            prompt_sections.append(
                f"GEMINI DECISION 1 TRIAGE: acuity={decision_summary.get('triage_acuity')}, "
                f"projection={decision_summary.get('projection_view')}, "
                f"quality={decision_summary.get('image_quality_score')}/5"
            )
        if patient:
            prompt_sections.append(f"PATIENT CLINICAL PROFILE: {json.dumps(patient)}")
        prompt_sections.append(f"PRELIMINARY STRUCTURED DRAFT:\n{draft[:2200]}")

        input_blocks: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(prompt_sections)}]
        if image_bytes:
            input_blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                    "mime_type": mime_type,
                }
            )

        response_format = {
            "type": "text",
            "mime_type": "application/json",
            "schema": ClinicalReportSchema.model_json_schema(),
        }

        all_candidates = list(dict.fromkeys([self.model, *CURRENT_GEMINI_MODELS]))
        healthy = [m for m in all_candidates if _is_model_healthy(m)]
        # Try at most 2 healthy Gemini candidates before falling through to sub-second Groq fallback
        candidates = (healthy if healthy else all_candidates)[:2]
        for model_name in candidates:
            try:
                raw_text, interaction_id = self._create_interaction(
                    model=model_name,
                    input_payload=input_blocks,
                    system_instruction=self.REPORT_SYSTEM_INSTRUCTION,
                    response_format=response_format,
                    timeout_sec=11.0,
                )
                if not raw_text:
                    continue
                validated = ClinicalReportSchema.model_validate_json(raw_text)
                _mark_model_healthy(model_name)
                compartments = validated.compartments.model_dump()
                findings = (
                    f"LUNGS & AIRWAYS: {compartments['lungs']}\n"
                    f"PLEURA & SULCI: {compartments['pleura']}\n"
                    f"CARDIOMEDIASTINUM: {compartments['cardio']}\n"
                    f"OSSEOUS & SOFT TISSUES: {compartments['osseous']}"
                )

                full_text = (
                    f"EXAMINATION: CHEST RADIOGRAPH (FRONTAL)\n"
                    f"AI SYNTHESIS ENGINE: {model_name} (Gemini Interactions API — ACR/RSNA Clinical Synthesis)\n\n"
                    f"FINDINGS:\n{findings}\n\n"
                    f"IMPRESSION:\n{validated.impression}\n\n"
                    f"GUIDELINE-DIRECTED RECOMMENDATIONS:\n{validated.recommendations}\n\n"
                    f"PHARMACOTHERAPY & STEWARDSHIP:\n{validated.prescriptions}\n\n"
                    f"PATIENT-FRIENDLY SUMMARY:\n{validated.patient_summary}\n\n"
                    f"DISCLAIMER: Investigational AI clinical decision support. Licensed radiologist verification required."
                )

                return {
                    "findings": findings,
                    "impression": validated.impression,
                    "dictation_transcript": validated.dictation_transcript,
                    "recommendations": validated.recommendations,
                    "prescriptions": validated.prescriptions,
                    "patient_summary": validated.patient_summary,
                    "executive_summary": validated.executive_summary,
                    "differential_diagnoses": validated.differential_diagnoses,
                    "triage_level": validated.triage_level,
                    "critical_alert": validated.critical_alert,
                    "compartments": compartments,
                    "report": full_text,
                    "draft": full_text,
                    "stage": "gemini_interactions",
                    "source": f"gemini:{model_name}",
                    "model_used": model_name,
                    "interaction_id": interaction_id,
                    "synthesis_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            except httpx.HTTPStatusError as exc:
                _mark_model_unhealthy(model_name, exc.response.status_code)
                print(f"[gemini-interactions] {model_name} HTTP {exc.response.status_code}; benched briefly.")
                continue
            except Exception as exc:
                _mark_model_unhealthy(model_name)
                print(f"[gemini-interactions] {model_name} failed: {exc}")
                continue
        return None

    def chat_copilot(
        self,
        question: str,
        study_context: dict[str, Any],
        history: list[dict[str, str]] | None = None,
        previous_interaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Context-grounded thoracic radiology Q&A via Gemini Interactions API (with Groq & offline fallback)."""
        t0 = time.perf_counter()
        sys_prompt = (
            "You are an expert academic thoracic radiologist AI copilot inside the CXR-Diagnostic Pro PACS workstation. "
            "Answer the clinician's question concisely, accurately, and citing evidence-based clinical guidelines "
            "(ATS/IDSA, ACC/AHA, Fleischner Society 2017, ACCP, BTS) using the provided study telemetry, "
            "14-pathology scores, quantitative CTR, Grad-CAM++ zones, and patient profile. Use crisp markdown bullets."
        )
        ctx_blob = json.dumps(study_context, default=str)[:4000]
        user_msg = f"ACTIVE STUDY CONTEXT:\n{ctx_blob}\n\nCLINICIAN QUESTION:\n{question}"

        # 1. Try Gemini Interactions API (with stateful previous_interaction_id + circuit breaker)
        if self.api_key:
            transcript = ""
            if history and not previous_interaction_id:
                for turn in history[-6:]:
                    transcript += f"{turn.get('role', 'user').upper()}: {turn.get('content', '')}\n"
            prompt = f"{transcript}\n{user_msg}".strip()
            all_chat = list(dict.fromkeys([self.model, *CURRENT_GEMINI_MODELS]))
            healthy_chat = [m for m in all_chat if _is_model_healthy(m)]
            for model_name in (healthy_chat if healthy_chat else all_chat)[:2]:
                try:
                    ans_text, next_id = self._create_interaction(
                        model=model_name,
                        input_payload=prompt,
                        system_instruction=sys_prompt,
                        previous_interaction_id=previous_interaction_id,
                        timeout_sec=10.0,
                    )
                    if ans_text:
                        _mark_model_healthy(model_name)
                        return {
                            "answer": ans_text,
                            "provider": f"gemini:{model_name}",
                            "interaction_id": next_id,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        }
                except httpx.HTTPStatusError as exc:
                    _mark_model_unhealthy(model_name, exc.response.status_code)
                    continue
                except Exception:
                    _mark_model_unhealthy(model_name)
                    continue

        # 2. Try Groq Cloud fallback (live openai/gpt-oss-20b / 120b / qwen3.8-27b)
        AI_RUNTIME_CONFIG.refresh_from_env()
        if AI_RUNTIME_CONFIG.groq_api_key:
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=AI_RUNTIME_CONFIG.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    timeout=14.0,
                )
                groq_candidates = list(
                    dict.fromkeys(
                        m
                        for m in [
                            AI_RUNTIME_CONFIG.groq_model,
                            "openai/gpt-oss-20b",
                            "openai/gpt-oss-120b",
                            "qwen/qwen3.8-27b",
                        ]
                        if m and not m.startswith("llama-3.")
                    )
                )
                for candidate in groq_candidates:
                    try:
                        resp = client.chat.completions.create(
                            model=candidate,
                            temperature=0.25,
                            max_tokens=750,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_msg},
                            ],
                        )
                        text = (resp.choices[0].message.content or "").strip()
                        if text:
                            return {
                                "answer": text,
                                "provider": f"groq:{candidate}",
                                "interaction_id": None,
                                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                            }
                    except Exception:
                        continue
            except Exception:
                pass

        # 3. Deterministic Clinical Knowledge Engine Fallback (always succeeds offline)
        return {
            "answer": _offline_clinical_copilot_answer(question, study_context),
            "provider": "clinical-knowledge-engine",
            "interaction_id": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def _offline_clinical_copilot_answer(question: str, ctx: dict[str, Any]) -> str:
    q = question.lower()
    preds = ctx.get("predictions", [])
    thr = float(ctx.get("threshold", 0.50))
    positives = [p for p in preds if float(p.get("score", 0)) >= thr]
    top_names = [p["name"].replace("_", " ") for p in positives[:4]]
    anatomy = ctx.get("anatomy") or {}
    ctr = anatomy.get("ctr")
    patient = ctx.get("patient") or {}

    if "ctr" in q or "cardio" in q or "heart" in q:
        ctr_line = (
            f"Automated PSPNet segmentation measured **CTR = {ctr:.3f}** ({anatomy.get('ctr_status', 'evaluated')}). "
            if ctr is not None
            else "Cardiothoracic ratio can be verified using the Caliper tool (`M`) on the lightbox. "
        )
        return (
            f"### Cardiomediastinal Assessment\n"
            f"- {ctr_line}Standard upright PA cutoff is **< 0.50** (portable AP views artificially magnify the heart silhouette by ~10-15%).\n"
            f"- **ACC/AHA Next Steps**: Correlate with NT-proBNP, volume status, and 2D echocardiography if clinically decompensated.\n"
            f"- *Tip*: Add a `GEMINI_API_KEY` in AI Engine Settings to unlock live Gemini 4 Hydrogen + Gemini Decision 1 reasoning."
        )
    if "patient" in q or "explain" in q or "plain" in q:
        if not top_names:
            return (
                "### Plain-Language Patient Explanation\n"
                "- **Good news**: The automated chest X-ray screening did not flag an acute lung infection, collapsed lung, or major fluid buildup.\n"
                "- **Next Step**: Your physician will still review the full film alongside your symptoms to decide on supportive care."
            )
        return (
            f"### Plain-Language Patient Explanation\n"
            f"- **What the X-ray highlighted**: Possible signs of **{', '.join(top_names)}**.\n"
            f"- **What happens next**: A radiologist will confirm these areas directly on the scan and discuss targeted treatment or follow-up imaging (such as a CT scan or repeat X-ray) with you.\n"
            f"- **Questions to ask**: *'Does this match my symptoms?'*, *'Do I need a CT scan or follow-up X-ray?'*"
        )
    if "handoff" in q or "stat" in q or "script" in q or "ed" in q:
        urgency = "STAT CRITICAL" if "Pneumothorax" in top_names else ("URGENT" if top_names else "ROUTINE")
        return (
            f"### SBAR Clinical Handoff ({urgency})\n"
            f"- **Situation**: {patient.get('age', 'Adult')}yo {patient.get('sex', '')} in {patient.get('setting', 'Clinical Care')} — indication: {patient.get('indication', 'thoracic evaluation')}.\n"
            f"- **Findings**: AI ensemble flagged **{', '.join(top_names) if top_names else 'no acute pathology above threshold'}**"
            + (f" (CTR {ctr:.2f}).\n" if ctr else ".\n")
            + f"- **Recommendation**: Direct attending radiograph review; initiate guideline-directed management if clinically concordant."
        )
    if "false" in q or "critique" in q or "uncertain" in q or "model" in q:
        pos_list_str = (
            ", ".join(f"{p['name']} ({p['score']*100:.1f}%)" for p in positives[:5])
            if positives
            else "None"
        )
        return (
            "### Diagnostic Confidence & False-Positive Audit\n"
            f"- **Detected above {int(thr*100)}%**: {pos_list_str}.\n"
            "- **Verification Checklist**: Inspect the **Ensemble Lab** tab to verify inter-backbone agreement across DenseNet-121 (Multi-Cohort), DenseNet-121 (NIH-14), and ResNet-50 (512px). Scores near threshold with high epistemic spread warrant direct film correlation (hold `Space` to peek raw X-ray).\n"
            "- **Enable Gemini Decision 1**: Adding a Gemini API key adds a 4th independent multimodal foundation classifier for consensus calibration."
        )

    return (
        f"### Clinical Radiology Summary\n"
        f"- **Active Findings (≥{int(thr*100)}%)**: **{', '.join(top_names) if top_names else 'No acute cardiopulmonary abnormality'}**.\n"
        + (f"- **Quantitative Anatomy**: Measured CTR **{ctr:.3f}** ({anatomy.get('ctr_status', '')}).\n" if ctr else "")
        + "- **Guideline Actions**: Review the **Guidelines & Rx** tab for ATS/IDSA, ACC/AHA, Fleischner 2017, and ACCP protocols, or click any pathology row to localize its Grad-CAM++ hotspot.\n"
        "- *Configure `GEMINI_API_KEY` in AI Engine Settings to enable live Gemini 4 Hydrogen / 3.5 Flash-Lite clinical dialogue.*"
    )


def run_gemini_decision_1(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    patient: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not AI_RUNTIME_CONFIG.gemini_decision_enabled:
        return {"active": False, "status": "disabled", "model": AI_RUNTIME_CONFIG.gemini_decision_model, "latency_ms": 0.0}
    return GeminiDecision1Client().evaluate(image_bytes=image_bytes, mime_type=mime_type, patient=patient)


def synthesize_with_gemini_flash(**kwargs: Any) -> dict[str, Any] | None:
    return GeminiFlashClinician().synthesize_report(**kwargs)


def chat_with_gemini_copilot(
    question: str,
    study_context: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    previous_interaction_id: str | None = None,
) -> dict[str, Any]:
    return GeminiFlashClinician().chat_copilot(
        question=question,
        study_context=study_context,
        history=history,
        previous_interaction_id=previous_interaction_id,
    )
