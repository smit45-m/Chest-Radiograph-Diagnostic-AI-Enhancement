"""Health-check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return model status, device info, and uptime."""
    state = request.app.state
    uptime = time.time() - getattr(state, "start_time", time.time())

    pipeline = getattr(state, "pipeline", None)
    model_loaded = pipeline is not None
    device = str(getattr(pipeline, "device", "unknown")) if model_loaded else "unknown"

    return HealthResponse(
        status="ok",
        device=device,
        model_loaded=model_loaded,
        uptime_seconds=round(uptime, 2),
    )
