"""Tests for CXR-Diagnostic Pro v4.0 upgrades:
- Asymmetric Loss & Focal Loss
- ThoracicTransferModel (GeM pooling + Multi-Sample Dropout) & TemperatureScaler
- Grad-CAM++ transparent RGBA PNG + hotspot bbox & anatomical zone localization
- Gemini Decision 1 Decisions API response parser + graceful no-key fallback
- Batch STAT Triage, Gemini Copilot, AI Config, and On-Demand Report Synthesis endpoints
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from cxr.ai.gemini_engine import GeminiDecision1Client
from cxr.constants import NUM_CLASSES, PATHOLOGIES
from cxr.interpret.gradcam import (
    GradCAMPlusPlus,
    cam_to_transparent_rgba_png,
    extract_hotspot_metadata,
)
from cxr.models.densenet import TemperatureScaler, ThoracicTransferModel
from cxr.models.losses import AsymmetricLossMultiLabel, build_loss


def test_asymmetric_loss_and_focal_loss():
    logits = torch.randn(4, NUM_CLASSES, requires_grad=True)
    targets = (torch.rand(4, NUM_CLASSES) > 0.85).float()

    asl = build_loss(loss_type="asl")
    assert isinstance(asl, AsymmetricLossMultiLabel)
    loss = asl(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

    focal = build_loss(loss_type="focal")
    loss_f = focal(logits.detach(), targets)
    assert torch.isfinite(loss_f)


def test_transfer_model_and_temperature_calibration():
    model = ThoracicTransferModel(arch="densenet121", pretrained=False, use_gem=True)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)

    scaler = TemperatureScaler(num_classes=NUM_CLASSES)
    val_logits = torch.randn(32, NUM_CLASSES) * 3.0
    val_targets = (torch.sigmoid(val_logits / 2.0) > 0.5).float()
    stats = scaler.fit_calibration(val_logits, val_targets, max_iter=20)
    assert stats["nll_after"] <= stats["nll_before"] + 1e-4


def test_gradcam_plusplus_and_hotspot_metadata(model, sample_image_tensor):
    with GradCAMPlusPlus(model, model.cam_target_layer()) as cam_engine:
        cam = cam_engine.generate(sample_image_tensor, class_idx=1, output_size=(128, 128))
    assert cam.shape == (128, 128)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5

    rgba_png = cam_to_transparent_rgba_png(cam, colormap="turbo")
    assert rgba_png[:4] == b"\x89PNG"

    meta = extract_hotspot_metadata(cam)
    assert 0.0 <= meta["peak_x"] <= 1.0
    assert 0.0 <= meta["peak_y"] <= 1.0
    assert len(meta["bbox"]) == 4
    assert isinstance(meta["anatomical_zone"], str) and len(meta["anatomical_zone"]) > 0


def test_gemini_decision_1_parser_and_offline_graceful():
    client = GeminiDecision1Client(api_key="")
    res_no_key = client.evaluate(b"fake_bytes")
    assert res_no_key["active"] is False
    assert res_no_key["status"] == "no_api_key"

    # Verify official v1beta Decisions API question schema (score criteria is ordered list!)
    qs = client.build_questions()
    assert len(qs) == 18
    assert qs["is_valid_cxr"]["type"] == "noul"
    assert qs["triage_acuity"]["type"] == "choice" and isinstance(qs["triage_acuity"]["criteria"], dict)
    assert qs["image_quality"]["type"] == "score" and isinstance(qs["image_quality"]["criteria"], list)
    assert len(qs["image_quality"]["criteria"]) == 5

    # Verify canonical multimodal input blocks
    blocks = client.build_multimodal_input(b"test-jpeg-bytes", mime_type="image/jpeg")
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image" and blocks[1]["mime_type"] == "image/jpeg"

    # Test structured parsing of a simulated Gemini Decision 1 v1beta response
    mock_answers = {
        "is_valid_cxr": {"noul": 0.98, "confidence": 0.96},
        "triage_acuity": {"choice": "STAT_CRITICAL", "probabilities": {"STAT_CRITICAL": 0.89}},
        "projection_view": {"choice": "PA_UPRIGHT"},
        "image_quality": {"score": 4.6},
        "path_Pneumothorax": {"noul": 0.91, "confidence": 0.93},
        "path_Cardiomegaly": {"noul": 0.12, "confidence": 0.88},
    }
    parsed = client._parse_decision_response(mock_answers, latency_ms=184.2, transport="rest")
    assert parsed["active"] is True
    assert parsed["output_tokens"] == 0
    assert parsed["is_valid_cxr"] is True
    assert parsed["triage_acuity"] == "STAT_CRITICAL"
    assert parsed["projection_view"] == "PA_UPRIGHT"
    assert parsed["image_quality_score"] == 4.6
    assert parsed["pathology_scores"]["Pneumothorax"] == pytest.approx(0.91, abs=1e-3)
    assert parsed["pathology_scores"]["Cardiomegaly"] == pytest.approx(0.12, abs=1e-3)


def test_v4_api_endpoints_batch_copilot_config_and_synthesize():
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    mock_pipe = MagicMock()
    mock_pipe.device = "cpu"
    mock_pipe.ensemble = None
    mock_pipe.predict.return_value = {
        "predictions": [
            {"name": "Pneumothorax", "score": 0.89, "severity": "severe"},
            *[{"name": p, "score": 0.08, "severity": "mild"} for p in PATHOLOGIES if p != "Pneumothorax"],
        ],
        "positives": [{"name": "Pneumothorax", "score": 0.89, "severity": "severe"}],
        "heatmaps": [],
        "anatomy": {"ctr": 0.46, "ctr_status": "Normal (<0.50)"},
        "report": {"triage_level": "STAT_CRITICAL", "critical_alert": True, "findings": "", "impression": "", "draft": "", "source": "template"},
        "inference_time_ms": 42.0,
        "vision_latency_ms": 42.0,
    }

    with TestClient(app) as client:
        app.state.pipeline = mock_pipe

        # 1. Batch triage endpoint
        res_batch = client.post(
            "/api/v1/predict/batch",
            files=[
                ("files", ("case_a.jpg", img_bytes, "image/jpeg")),
                ("files", ("case_b.jpg", img_bytes, "image/jpeg")),
            ],
            data={"threshold": "0.50"},
        )
        assert res_batch.status_code == 200
        bdata = res_batch.json()
        assert bdata["total_studies"] == 2
        assert bdata["stat_count"] == 2

        # 2. AI Config GET & POST
        res_cfg = client.get("/api/v1/ai/config")
        assert res_cfg.status_code == 200
        assert "gemini_flash_model" in res_cfg.json()

        res_upd = client.post(
            "/api/v1/ai/config",
            json={"gemini_flash_model": "gemini-4-hydrogen", "decision_fusion_weight": 0.35},
        )
        assert res_upd.status_code == 200
        assert res_upd.json()["gemini_flash_model"] == "gemini-4-hydrogen"
        assert res_upd.json()["decision_fusion_weight"] == pytest.approx(0.35)

        # 3. Copilot Chat endpoint (offline fallback always succeeds cleanly)
        res_chat = client.post(
            "/api/v1/copilot/chat",
            json={
                "question": "Interpret the measured CTR and cardiac risk",
                "study_context": {"threshold": 0.5, "predictions": [], "anatomy": {"ctr": 0.54, "ctr_status": "Enlarged"}},
            },
        )
        assert res_chat.status_code == 200
        assert "answer" in res_chat.json() and len(res_chat.json()["answer"]) > 20

        # 4. On-demand report synthesis endpoint
        scores = {p: 0.05 for p in PATHOLOGIES}
        scores["Cardiomegaly"] = 0.84
        res_rep = client.post(
            "/api/v1/report/synthesize",
            json={"scores": scores, "threshold": 0.5, "use_llm": False},
        )
        assert res_rep.status_code == 200
        assert "Cardiomegaly" in res_rep.json()["impression"]


def test_v9_single_pass_gradcam_pathology_zone_and_streaming(model, sample_image_tensor):
    """Verify v9.0 upgrades:
    1. Single-pass multi-class Grad-CAM++ produces valid maps for multiple classes in 1 forward pass.
    2. Pathology-aware anatomical zone prioritizes lung parenchyma for pulmonary diseases.
    3. Projection-aware CTR interpretation adjusts cutoff for AP portable scans.
    4. POST /api/v1/predict/stream returns valid NDJSON milestones ending in `complete`.
    """
    import json

    from cxr.models.ensemble import ClinicalEnsemble

    # 1. Single-pass multi-class Grad-CAM++
    with GradCAMPlusPlus(model, model.cam_target_layer()) as cam_engine:
        cams = cam_engine.generate_multi(sample_image_tensor, class_indices=[0, 1, 3], output_size=(64, 64))
    assert set(cams.keys()) == {0, 1, 3}
    for c in cams.values():
        assert c.shape == (64, 64)
        assert c.min() >= 0.0 and c.max() <= 1.0 + 1e-5

    # 2. Pathology-aware anatomical zone (lung overlap wins for Infiltration even with mediastinal spillover)
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[24:44, 16:40] = 0.9
    seg_masks = {
        "Right Lung": np.zeros((64, 64), dtype=bool),
        "Left Lung": np.zeros((64, 64), dtype=bool),
        "Heart": np.zeros((64, 64), dtype=bool),
    }
    seg_masks["Right Lung"][18:48, 12:30] = True
    seg_masks["Heart"][28:46, 26:42] = True
    lung_meta = extract_hotspot_metadata(cam, seg_masks=seg_masks, pathology="Infiltration")
    cardiac_meta = extract_hotspot_metadata(cam, seg_masks=seg_masks, pathology="Cardiomegaly")
    assert "Right Lung" in lung_meta["anatomical_zone"]
    assert cardiac_meta["anatomical_zone"] == "Cardiomediastinal Silhouette"

    # 3. Projection-aware CTR interpretation
    pa_status, pa_cut = ClinicalEnsemble.interpret_ctr(0.54, "PA_UPRIGHT")
    ap_status, ap_cut = ClinicalEnsemble.interpret_ctr(0.54, "AP_PORTABLE")
    assert pa_cut == pytest.approx(0.53) and "Enlarged" in pa_status
    assert ap_cut == pytest.approx(0.58) and "Normal for AP" in ap_status

    # 4. NDJSON Streaming endpoint
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    mock_pipe = MagicMock()
    mock_pipe.device = "cpu"
    mock_pipe.predict_stream.return_value = iter([
        {
            "event": "vision",
            "data": {
                "predictions": [{"name": p, "score": 0.1, "severity": "mild", "review_flag": "clear"} for p in PATHOLOGIES],
                "heatmaps": [],
                "vision_latency_ms": 180.0,
            },
        },
        {
            "event": "complete",
            "data": {
                "predictions": [{"name": p, "score": 0.1, "severity": "mild", "review_flag": "clear"} for p in PATHOLOGIES],
                "positives": [],
                "heatmaps": [],
                "anatomy": {"ctr": 0.45, "ctr_status": "Normal (<0.50)", "ctr_cutoff": 0.53, "projection_view": "PA_UPRIGHT"},
                "report": {"findings": "Clear", "impression": "Normal", "draft": "Normal", "source": "template"},
                "cache_hit": False,
                "inference_time_ms": 210.0,
                "vision_latency_ms": 180.0,
            },
        },
    ])

    with TestClient(app) as client:
        app.state.pipeline = mock_pipe
        res = client.post(
            "/api/v1/predict/stream",
            files={"file": ("xray.jpg", img_bytes, "image/jpeg")},
            data={"threshold": "0.50"},
        )
        assert res.status_code == 200
        events = [json.loads(line) for line in res.text.strip().splitlines() if line.strip()]
        assert [e["event"] for e in events] == ["vision", "complete"]
        assert events[-1]["data"]["anatomy"]["ctr_cutoff"] == pytest.approx(0.53)
