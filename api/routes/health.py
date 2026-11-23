"""Health-check & system capabilities endpoint."""

from __future__ import annotations

import time

import torch
from fastapi import APIRouter, Request

from api.schemas import HealthResponse
from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return model ensemble status, GPU telemetry, Gemini configuration, and uptime."""
    state = request.app.state
    uptime = time.time() - getattr(state, "start_time", time.time())

    pipeline = getattr(state, "pipeline", None)
    model_loaded = pipeline is not None
    device = str(getattr(pipeline, "device", "unknown")) if model_loaded else "unknown"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    ensemble = getattr(pipeline, "ensemble", None) if model_loaded else None
    active_backbones = list(getattr(ensemble, "active_backbones", [])) if ensemble is not None else (
        ["densenet121_local"] if model_loaded else []
    )
    seg_ready = bool(getattr(ensemble, "seg_model", None) is not None) if ensemble is not None else False
    batcher = getattr(state, "batcher", None)
    ai_status = AI_RUNTIME_CONFIG.public_status()

    return HealthResponse(
        status="ok",
        device=device,
        gpu_name=gpu_name,
        model_loaded=model_loaded,
        active_backbones=active_backbones,
        segmentation_ready=seg_ready,
        gemini_configured=ai_status["gemini_configured"],
        gemini_flash_model=ai_status["gemini_flash_model"],
        gemini_decision_enabled=ai_status["gemini_decision_enabled"],
        batcher_active=batcher is not None,
        uptime_seconds=round(uptime, 2),
    )
