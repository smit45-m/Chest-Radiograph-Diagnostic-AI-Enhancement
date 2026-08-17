"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from cxr.constants import PATHOLOGIES


@pytest.fixture
def dummy_image_bytes() -> bytes:
    """Create a 224x224 grayscale JPEG in memory."""
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_pipeline():
    """Mocked DiagnosisPipeline returning valid inference structure."""
    pipeline = MagicMock()
    pipeline.device = "cpu"
    pipeline.predict.return_value = {
        "predictions": [
            {"name": name, "score": 0.1, "severity": "mild"} for name in PATHOLOGIES
        ],
        "positives": [
            {"name": "Cardiomegaly", "score": 0.88, "severity": "severe"}
        ],
        "heatmaps": [
            {"pathology": "Cardiomegaly", "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}
        ],
        "report": {
            "findings": "The cardiomediastinal silhouette is enlarged.",
            "impression": "Cardiomegaly present.",
            "draft": "EXAMINATION: CHEST RADIOGRAPH\nFINDINGS: The cardiomediastinal silhouette is enlarged.\nIMPRESSION: Cardiomegaly present.",
            "source": "template",
        },
        "inference_time_ms": 12.5,
    }
    return pipeline


def test_root_frontend():
    """Bare root / should serve the interactive PACS Diagnostic frontend HTML."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "CXR-Diagnostic" in response.text


def test_health_endpoint(mock_pipeline):
    """GET /api/v1/health returns status ok."""
    with TestClient(app) as client:
        app.state.pipeline = mock_pipeline
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data


def test_predict_endpoint_success(mock_pipeline, dummy_image_bytes):
    """POST /api/v1/predict returns prediction response."""
    with TestClient(app) as client:
        app.state.pipeline = mock_pipeline
        response = client.post(
            "/api/v1/predict",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
            data={"threshold": "0.45"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(PATHOLOGIES)
        assert "heatmaps" in data
        assert "report" in data
        assert data["inference_time_ms"] > 0


def test_predict_no_file():
    """POST /api/v1/predict without file returns 422 Unprocessable Entity."""
    with TestClient(app) as client:
        response = client.post("/api/v1/predict")
        assert response.status_code == 422


def test_predict_invalid_image(mock_pipeline):
    """POST /api/v1/predict with corrupt data returns 400 Bad Request."""
    with TestClient(app) as client:
        app.state.pipeline = mock_pipeline
        response = client.post(
            "/api/v1/predict",
            files={"file": ("corrupt.txt", b"not-an-image-data", "text/plain")},
            data={"threshold": "0.45"},
        )
        assert response.status_code == 400
