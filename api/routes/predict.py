"""Prediction endpoints — single image and batch."""

from __future__ import annotations

import asyncio
import io
import time
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from api.schemas import (
    HeatmapData,
    PathologyPrediction,
    PredictionResponse,
    ReportData,
)

router = APIRouter(prefix="/api/v1", tags=["predict"])


async def _run_pipeline(
    request: Request,
    image_bytes: bytes,
    threshold: float,
) -> PredictionResponse:
    """Shared logic for single-image prediction."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    # Validate image
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid image: {exc}"
        ) from exc

    # Run inference in a thread so the event loop stays free.
    try:
        result = await asyncio.to_thread(
            pipeline.predict, image_bytes, threshold
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    # Map result dicts → Pydantic models.
    predictions = [
        PathologyPrediction(**p) for p in result.get("predictions", [])
    ]
    heatmaps = [HeatmapData(**h) for h in result.get("heatmaps", [])]

    report_raw = result.get("report", {})
    report = ReportData(
        findings=report_raw.get("findings", ""),
        impression=report_raw.get("impression", ""),
        full_text=report_raw.get("report", report_raw.get("draft", "")),
        source=report_raw.get("source", "template"),
    )

    return PredictionResponse(
        predictions=predictions,
        heatmaps=heatmaps,
        report=report,
        inference_time_ms=result.get("inference_time_ms", 0.0),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)"),
    threshold: float = Form(0.45, description="Pathology detection threshold"),
) -> PredictionResponse:
    """Upload a chest X-ray and receive pathology predictions, Grad-CAM
    heatmaps, and an auto-generated radiology report."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    return await _run_pipeline(request, image_bytes, threshold)


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Multiple chest X-ray images"),
    threshold: float = Form(0.45, description="Pathology detection threshold"),
) -> List[PredictionResponse]:
    """Upload multiple images for batch prediction."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tasks = []
    for f in files:
        raw = await f.read()
        if raw:
            tasks.append(_run_pipeline(request, raw, threshold))

    if not tasks:
        raise HTTPException(status_code=400, detail="All uploaded files are empty")

    return await asyncio.gather(*tasks)
