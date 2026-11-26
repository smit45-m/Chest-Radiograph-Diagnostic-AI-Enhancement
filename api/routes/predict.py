"""Prediction, Batch Triage, Grad-CAM++, Report Synthesis, Gemini Copilot & AI Config endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from api.schemas import (
    AIConfigUpdate,
    AnatomyData,
    BatchPredictionResponse,
    BatchStudyItem,
    CopilotChatRequest,
    CopilotChatResponse,
    DecisionSummary,
    HeatmapData,
    PathologyPrediction,
    PatientProfile,
    PredictionResponse,
    ReportData,
    ReportSynthesizeRequest,
    SampleCaseResponse,
    SingleHeatmapResponse,
)
from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG, chat_with_gemini_copilot
from cxr.db.store import CLINICAL_DB
from cxr.reports.generator import generate_report
from cxr.telemetry import TELEMETRY_HUB

router = APIRouter(prefix="/api/v1", tags=["predict"])

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_test_set"


def _normalize_and_validate_image_bytes(image_bytes: bytes) -> tuple[bytes, dict[str, Any] | None]:
    """Validate image bytes, convert DICOM (.dcm) to PNG, and extract clinical DICOM tags."""
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # 1. Standard PIL image formats (JPEG, PNG, WEBP, TIFF, BMP)
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.verify()
        return image_bytes, None
    except Exception:
        pass

    # 2. DICOM medical imaging format (.dcm)
    try:
        import numpy as np
        import pydicom

        dcm = pydicom.dcmread(io.BytesIO(image_bytes))
        dicom_meta: dict[str, Any] = {
            "modality": str(getattr(dcm, "Modality", "CR/DX")),
            "view_position": str(getattr(dcm, "ViewPosition", "Unknown")),
            "patient_sex": str(getattr(dcm, "PatientSex", "")),
            "patient_age": str(getattr(dcm, "PatientAge", "")),
            "study_description": str(getattr(dcm, "StudyDescription", "Chest Radiograph")),
            "manufacturer": str(getattr(dcm, "Manufacturer", "")),
            "kvp": str(getattr(dcm, "KVP", "")),
            "rows": int(getattr(dcm, "Rows", 0)),
            "columns": int(getattr(dcm, "Columns", 0)),
            "photometric_interpretation": str(getattr(dcm, "PhotometricInterpretation", "")),
        }

        pixel_array = dcm.pixel_array.astype(float)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        pixel_array = pixel_array * slope + intercept

        if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
            wc = dcm.WindowCenter
            ww = dcm.WindowWidth
            wc = float(wc[0]) if isinstance(wc, (list, pydicom.multival.MultiValue)) else float(wc)
            ww = float(ww[0]) if isinstance(ww, (list, pydicom.multival.MultiValue)) else float(ww)
            min_val = wc - ww / 2.0
            max_val = wc + ww / 2.0
            clipped = np.clip(pixel_array, min_val, max_val)
            normalized = ((clipped - min_val) / max(max_val - min_val, 1e-6) * 255.0).astype(np.uint8)
        else:
            p_min, p_max = pixel_array.min(), pixel_array.max()
            if p_max > p_min:
                normalized = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros(pixel_array.shape, dtype=np.uint8)

        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            normalized = 255 - normalized

        out_img = Image.fromarray(normalized).convert("RGB")
        out_buf = io.BytesIO()
        out_img.save(out_buf, format="PNG")
        return out_buf.getvalue(), dicom_meta
    except Exception as dcm_exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a supported image format. Upload a chest radiograph (JPEG, PNG, WEBP, TIFF, or DICOM .dcm).",
        ) from dcm_exc


def _map_report_schema(report_raw: dict[str, Any]) -> ReportData:
    return ReportData(
        findings=report_raw.get("findings", ""),
        impression=report_raw.get("impression", ""),
        dictation_transcript=report_raw.get("dictation_transcript", ""),
        recommendations=report_raw.get("recommendations", ""),
        prescriptions=report_raw.get("prescriptions", ""),
        patient_summary=report_raw.get("patient_summary", ""),
        executive_summary=report_raw.get("executive_summary", []),
        differential_diagnoses=report_raw.get("differential_diagnoses", []),
        triage_level=report_raw.get("triage_level", "ROUTINE"),
        critical_alert=bool(report_raw.get("critical_alert", False)),
        compartments=report_raw.get("compartments", {}),
        full_text=report_raw.get("report", report_raw.get("draft", "")),
        source=report_raw.get("source", report_raw.get("stage", "template")),
    )


def _enrich_patient_from_dicom(
    patient_dict: dict[str, Any] | None,
    dicom_meta: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Extract DICOM ViewPosition and fill missing patient age/sex automatically from DICOM headers."""
    if not dicom_meta:
        return patient_dict, None
    out = dict(patient_dict) if patient_dict else {}
    raw_age = str(dicom_meta.get("patient_age", "") or "").strip().upper()
    if not out.get("age") and raw_age:
        digits = "".join(ch for ch in raw_age if ch.isdigit())
        if digits:
            val = int(digits)
            if 1 <= val <= 120:
                out["age"] = val
    raw_sex = str(dicom_meta.get("patient_sex", "") or "").strip().upper()
    if (not out.get("sex") or out.get("sex") == "Unspecified") and raw_sex in ("M", "F"):
        out["sex"] = "Male" if raw_sex == "M" else "Female"
    view_pos = str(dicom_meta.get("view_position", "") or "").strip()
    dicom_view = view_pos if view_pos and view_pos.lower() != "unknown" else None
    return (out if out else None), dicom_view


def _build_prediction_response(
    result: dict[str, Any],
    elapsed_ms: float,
    dicom_meta: dict[str, Any] | None,
    patient_dict: dict[str, Any] | None,
    saved_study_id: str | None,
    saved_user_id: str | None,
) -> PredictionResponse:
    vision_ms = float(result.get("vision_latency_ms", elapsed_ms))
    predictions = [PathologyPrediction(**p) for p in result.get("predictions", [])]
    heatmaps = [HeatmapData(**h) for h in result.get("heatmaps", [])]
    report = _map_report_schema(result.get("report", {}))
    anatomy_raw = result.get("anatomy")
    anatomy = AnatomyData(**anatomy_raw) if isinstance(anatomy_raw, dict) and anatomy_raw else None
    dec_raw = result.get("decision_summary")
    decision_summary = DecisionSummary(**dec_raw) if isinstance(dec_raw, dict) and dec_raw else None
    patient_profile_obj = PatientProfile(**patient_dict) if patient_dict else None
    return PredictionResponse(
        study_id=saved_study_id,
        user_id=saved_user_id,
        predictions=predictions,
        heatmaps=heatmaps,
        anatomy=anatomy,
        decision_summary=decision_summary,
        ensemble_info=result.get("ensemble_info", {}),
        timings=result.get("timings", {}),
        dicom_metadata=dicom_meta,
        report=report,
        patient=patient_profile_obj,
        cache_hit=bool(result.get("cache_hit", False)),
        inference_time_ms=elapsed_ms,
        vision_latency_ms=vision_ms,
    )


async def _archive_study_if_requested(
    user_id: str | None,
    study_name: str | None,
    filename: str | None,
    sample_id: str | None,
    result: dict[str, Any],
    elapsed_ms: float,
) -> tuple[str | None, str | None]:
    if not user_id:
        return None, None
    try:
        vision_ms = float(result.get("vision_latency_ms", elapsed_ms))
        report = _map_report_schema(result.get("report", {}))
        anatomy_raw = result.get("anatomy")
        anatomy = AnatomyData(**anatomy_raw) if isinstance(anatomy_raw, dict) and anatomy_raw else None
        dec_raw = result.get("decision_summary")
        decision_summary = DecisionSummary(**dec_raw) if isinstance(dec_raw, dict) and dec_raw else None
        predictions = [PathologyPrediction(**p) for p in result.get("predictions", [])]
        saved = await asyncio.to_thread(
            CLINICAL_DB.save_study,
            user_id,
            {
                "study_name": study_name or (filename or "Chest Radiograph"),
                "filename": filename or "",
                "sample_id": sample_id,
                "predictions": [p.model_dump() for p in predictions],
                "anatomy": anatomy.model_dump() if anatomy else None,
                "decision_summary": decision_summary.model_dump() if decision_summary else None,
                "report": report.model_dump(),
                "ensemble_info": result.get("ensemble_info", {}),
                "timings": result.get("timings", {}),
                "vision_latency_ms": vision_ms,
                "total_latency_ms": elapsed_ms,
            },
        )
        return saved.get("id"), saved.get("user_id")
    except Exception as db_exc:
        print(f"[db] Non-fatal study archive error: {db_exc}")
        return None, user_id


async def _run_pipeline(
    request: Request,
    image_bytes: bytes,
    threshold: float,
    tta: bool = False,
    patient_dict: dict[str, Any] | None = None,
    use_gemini_decision: bool = True,
    use_llm: bool = True,
    colormap: str = "turbo",
    run_segmentation: bool = True,
    user_id: str | None = None,
    study_name: str | None = None,
    filename: str | None = None,
    sample_id: str | None = None,
) -> PredictionResponse:
    """Shared logic for single-image multi-backbone prediction & clinical report synthesis."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    valid_bytes, dicom_meta = _normalize_and_validate_image_bytes(image_bytes)
    patient_dict, dicom_view = _enrich_patient_from_dicom(patient_dict, dicom_meta)

    # Run inference in worker thread so FastAPI event loop stays fully non-blocking
    t_start = time.perf_counter()
    try:
        # Backward-compatible signature invocation (works with mocks in tests/test_api.py!)
        import inspect

        sig = inspect.signature(pipeline.predict)
        kwargs: dict[str, Any] = {}
        if "use_gemini_decision" in sig.parameters:
            kwargs["use_gemini_decision"] = use_gemini_decision
        if "use_llm" in sig.parameters:
            kwargs["use_llm"] = use_llm
        if "colormap" in sig.parameters:
            kwargs["colormap"] = colormap
        if "run_segmentation" in sig.parameters:
            kwargs["run_segmentation"] = run_segmentation
        if "dicom_view" in sig.parameters and dicom_view is not None:
            kwargs["dicom_view"] = dicom_view

        result = await asyncio.to_thread(
            pipeline.predict, valid_bytes, threshold, tta, patient_dict, **kwargs
        )
    except Exception as exc:
        TELEMETRY_HUB.record_error(
            error_type="PredictionError",
            message=str(exc),
            context={"threshold": threshold, "tta": tta},
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    elapsed_ms = float(result.get("inference_time_ms", (time.perf_counter() - t_start) * 1000))
    vision_ms = float(result.get("vision_latency_ms", elapsed_ms))

    TELEMETRY_HUB.record_span(
        op="cxr.predict.pipeline",
        description=f"Ensemble + Grad-CAM++ + Report (thr={threshold:.2f}, tta={tta}, vis={vision_ms:.1f}ms, cache={bool(result.get('cache_hit'))})",
        duration_ms=vision_ms,
        status="ok",
        tags={"threshold": threshold, "tta": tta, "vision_ms": vision_ms, "total_ms": elapsed_ms},
    )

    saved_study_id, saved_user_id = await _archive_study_if_requested(
        user_id, study_name, filename, sample_id, result, elapsed_ms
    )
    return _build_prediction_response(
        result, elapsed_ms, dicom_meta, patient_dict, saved_study_id, saved_user_id
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG/WEBP/DICOM)"),
    threshold: float = Form(0.50, description="Pathology detection threshold"),
    tta: bool = Form(True, description="Enable fast 2-view non-chiral Test-Time Augmentation"),
    use_gemini_decision: bool = Form(True, description="Enable Gemini Decision 1 multimodal consensus"),
    use_llm: bool = Form(True, description="Enable Gemini Flash / Groq clinical report synthesis"),
    colormap: str = Form("turbo", description="Grad-CAM++ colormap (turbo/inferno/viridis/clinical/jet)"),
    run_segmentation: bool = Form(True, description="Run PSPNet anatomical lung/heart CTR segmentation"),
    user_id: str | None = Form(None, description="Optional persisted user/patient ID to archive this study"),
    study_name: str | None = Form(None, description="Optional human-readable study label"),
    sample_id: str | None = Form(None, description="Optional benchmark sample ID"),
    patient_context: str | None = Form(None, description="Optional JSON string of patient profile"),
    patient_age: int | None = Form(None),
    patient_sex: str | None = Form(None),
    patient_setting: str | None = Form(None),
    patient_indication: str | None = Form(None),
    patient_comorbidities: str | None = Form(None),
    patient_smoking: str | None = Form(None),
) -> PredictionResponse:
    """Upload a chest X-ray and receive multi-backbone predictions, PSPNet CTR,
    Grad-CAM++ overlays, Gemini Decision 1 consensus, and Gemini Flash clinical report."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    patient_dict: dict[str, Any] = {}
    if patient_context:
        try:
            patient_dict = json.loads(patient_context)
        except Exception:
            pass

    for key, val in (
        ("age", patient_age),
        ("sex", patient_sex),
        ("setting", patient_setting),
        ("indication", patient_indication),
        ("comorbidities", patient_comorbidities),
        ("smoking", patient_smoking),
    ):
        if val is not None:
            patient_dict[key] = val

    return await _run_pipeline(
        request,
        image_bytes,
        threshold,
        tta=tta,
        patient_dict=patient_dict if patient_dict else None,
        use_gemini_decision=use_gemini_decision,
        use_llm=use_llm,
        colormap=colormap,
        run_segmentation=run_segmentation,
        user_id=user_id,
        study_name=study_name,
        filename=file.filename,
        sample_id=sample_id,
    )


@router.post("/predict/stream")
async def predict_stream(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG/WEBP/DICOM)"),
    threshold: float = Form(0.50),
    tta: bool = Form(True),
    use_gemini_decision: bool = Form(True),
    use_llm: bool = Form(True),
    colormap: str = Form("turbo"),
    run_segmentation: bool = Form(True),
    user_id: str | None = Form(None),
    study_name: str | None = Form(None),
    sample_id: str | None = Form(None),
    patient_context: str | None = Form(None),
) -> StreamingResponse:
    """Progressive NDJSON streaming endpoint (`vision` ~350ms -> `consensus` ~1.2s -> `complete`)."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    valid_bytes, dicom_meta = _normalize_and_validate_image_bytes(image_bytes)
    patient_dict: dict[str, Any] = {}
    if patient_context:
        try:
            patient_dict = json.loads(patient_context)
        except Exception:
            pass
    patient_dict, dicom_view = _enrich_patient_from_dicom(patient_dict if patient_dict else None, dicom_meta)
    filename = file.filename

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def _worker() -> None:
        t_start = time.perf_counter()
        try:
            if hasattr(pipeline, "predict_stream") and callable(pipeline.predict_stream):
                for evt in pipeline.predict_stream(
                    image_input=valid_bytes,
                    threshold=threshold,
                    tta=tta,
                    patient=patient_dict,
                    use_gemini_decision=use_gemini_decision,
                    use_llm=use_llm,
                    colormap=colormap,
                    run_segmentation=run_segmentation,
                    dicom_view=dicom_view,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", evt))
            else:
                res = pipeline.predict(valid_bytes, threshold, tta, patient_dict)
                res.setdefault("inference_time_ms", round((time.perf_counter() - t_start) * 1000, 2))
                loop.call_soon_threadsafe(queue.put_nowait, ("event", {"event": "complete", "data": res}))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    worker_task = asyncio.create_task(asyncio.to_thread(_worker))

    async def _ndjson_gen():
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    yield json.dumps({"event": "error", "detail": payload}) + "\n"
                    break
                evt_name = payload.get("event")
                evt_data = payload.get("data", {})
                if evt_name == "complete":
                    elapsed_ms = float(evt_data.get("inference_time_ms", 0.0))
                    vision_ms = float(evt_data.get("vision_latency_ms", elapsed_ms))
                    TELEMETRY_HUB.record_span(
                        op="cxr.predict.stream",
                        description=f"Streaming Ensemble (thr={threshold:.2f}, vis={vision_ms:.1f}ms, cache={bool(evt_data.get('cache_hit'))})",
                        duration_ms=vision_ms,
                        status="ok",
                        tags={"threshold": threshold, "vision_ms": vision_ms, "total_ms": elapsed_ms},
                    )
                    saved_study_id, saved_user_id = await _archive_study_if_requested(
                        user_id, study_name, filename, sample_id, evt_data, elapsed_ms
                    )
                    resp_obj = _build_prediction_response(
                        evt_data, elapsed_ms, dicom_meta, patient_dict, saved_study_id, saved_user_id
                    )
                    yield json.dumps({"event": "complete", "data": resp_obj.model_dump()}) + "\n"
                else:
                    if dicom_meta:
                        evt_data["dicom_metadata"] = dicom_meta
                    if patient_dict:
                        evt_data["patient"] = patient_dict
                    yield json.dumps({"event": evt_name, "data": evt_data}, default=str) + "\n"
        finally:
            await worker_task

    return StreamingResponse(
        _ndjson_gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Multiple chest X-ray radiographs for STAT triage"),
    threshold: float = Form(0.50),
    tta: bool = Form(False),
) -> BatchPredictionResponse:
    """Concurrent multi-study triage queue with automatic STAT-first clinical prioritization."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch triage")

    import inspect

    sig = inspect.signature(pipeline.predict)
    batch_kwargs: dict[str, Any] = {
        "use_gemini_decision": False,
        "use_llm": False,
        "run_segmentation": True,
    }
    if "generate_heatmaps" in sig.parameters:
        # Batch STAT triage never renders individual heatmaps — skipping cuts batch latency by ~2.5x!
        batch_kwargs["generate_heatmaps"] = False

    t0 = time.perf_counter()
    items: list[BatchStudyItem] = []

    for f in files[:24]:
        raw = await f.read()
        fname = f.filename or "radiograph.png"
        try:
            valid_bytes, dicom_meta = _normalize_and_validate_image_bytes(raw)
            _, dicom_view = _enrich_patient_from_dicom(None, dicom_meta)
            call_kw = dict(batch_kwargs)
            if "dicom_view" in sig.parameters and dicom_view is not None:
                call_kw["dicom_view"] = dicom_view
            res = await asyncio.to_thread(
                pipeline.predict,
                valid_bytes,
                threshold,
                tta,
                None,
                **call_kw,
            )
            preds = [PathologyPrediction(**p) for p in res.get("predictions", [])]
            pos = [p for p in preds if p.score >= threshold]
            rep = res.get("report", {})
            triage = rep.get("triage_level", "ROUTINE")
            crit = bool(rep.get("critical_alert", False)) or triage == "STAT_CRITICAL"
            ctr = (res.get("anatomy") or {}).get("ctr")
            items.append(
                BatchStudyItem(
                    filename=fname,
                    triage_level=triage,
                    critical_alert=crit,
                    top_findings=pos[:4] if pos else preds[:2],
                    ctr=ctr,
                    vision_latency_ms=float(res.get("vision_latency_ms", res.get("inference_time_ms", 0.0))),
                )
            )
        except Exception as exc:
            items.append(
                BatchStudyItem(
                    filename=fname,
                    triage_level="ROUTINE",
                    critical_alert=False,
                    top_findings=[],
                    vision_latency_ms=0.0,
                    error=str(exc),
                )
            )

    # Automatically sort worklist STAT_CRITICAL -> URGENT -> SEMI_URGENT -> ROUTINE
    priority = {"STAT_CRITICAL": 0, "URGENT": 1, "SEMI_URGENT": 2, "ROUTINE": 3}
    items.sort(key=lambda x: (priority.get(x.triage_level, 9), -(x.top_findings[0].score if x.top_findings else 0.0)))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    stat_count = sum(1 for it in items if it.critical_alert or it.triage_level == "STAT_CRITICAL")
    TELEMETRY_HUB.record_span(
        op="cxr.predict.batch",
        description=f"Batch Triage ({len(items)} studies, {stat_count} STAT)",
        duration_ms=total_ms,
        status="ok",
        tags={"batch_size": len(items), "stat_count": stat_count},
    )
    return BatchPredictionResponse(
        studies=items,
        total_studies=len(items),
        total_latency_ms=total_ms,
        avg_latency_ms=round(total_ms / max(len(items), 1), 2),
        stat_count=stat_count,
    )


@router.post("/predict/heatmap", response_model=SingleHeatmapResponse)
async def predict_single_heatmap(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image"),
    pathology: str = Form(..., description="Pathology to localize, e.g. Pneumonia"),
    colormap: str = Form("turbo", description="Colormap: turbo / inferno / viridis / clinical / jet"),
) -> SingleHeatmapResponse:
    """Generate an on-demand Grad-CAM++ heatmap & anatomical zone for a specific pathology."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    valid_bytes, _ = _normalize_and_validate_image_bytes(image_bytes)

    import inspect

    sig = inspect.signature(pipeline.generate_single_heatmap)
    if "colormap" in sig.parameters:
        res = await asyncio.to_thread(pipeline.generate_single_heatmap, valid_bytes, pathology, colormap)
    else:
        res = await asyncio.to_thread(pipeline.generate_single_heatmap, valid_bytes, pathology)

    if not res:
        raise HTTPException(status_code=400, detail=f"Could not generate heatmap for {pathology}")

    return SingleHeatmapResponse(**res)


@router.post("/report/synthesize", response_model=ReportData)
async def synthesize_report_endpoint(payload: ReportSynthesizeRequest) -> ReportData:
    """Re-synthesize the clinical report on demand (e.g. after updating patient clinical context)."""
    t0 = time.perf_counter()
    patient_dict = payload.patient.model_dump() if payload.patient else None
    raw = await asyncio.to_thread(
        generate_report,
        scores=payload.scores,
        threshold=payload.threshold,
        use_t5=False,
        use_llm=payload.use_llm,
        patient=patient_dict,
        anatomy=payload.anatomy,
        hotspots=payload.hotspots,
    )
    TELEMETRY_HUB.record_span(
        op="cxr.report.synthesize",
        description=f"Clinical Report Synthesis ({raw.get('source', 'template')})",
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status="ok",
        tags={"source": str(raw.get("source", "template"))},
    )
    return _map_report_schema(raw)


@router.post("/copilot/chat", response_model=CopilotChatResponse)
async def copilot_chat(payload: CopilotChatRequest) -> CopilotChatResponse:
    """Interactive Gemini Interactions API thoracic radiology copilot grounded in the active study."""
    res = await asyncio.to_thread(
        chat_with_gemini_copilot,
        question=payload.question,
        study_context=payload.study_context,
        history=payload.history,
        previous_interaction_id=payload.previous_interaction_id,
    )
    # Persist both clinician question & real AI response into SQLite if linked to a saved study
    if payload.study_id and payload.user_id:
        try:
            await asyncio.to_thread(
                CLINICAL_DB.add_copilot_message,
                payload.study_id,
                payload.user_id,
                "user",
                payload.question,
            )
            await asyncio.to_thread(
                CLINICAL_DB.add_copilot_message,
                payload.study_id,
                payload.user_id,
                "assistant",
                res["answer"],
                res.get("provider"),
                res.get("interaction_id"),
                float(res["latency_ms"]),
            )
        except Exception as db_exc:
            print(f"[db] Non-fatal copilot archive error: {db_exc}")

    TELEMETRY_HUB.record_span(
        op="cxr.copilot.chat",
        description=f"Radiology Copilot ({res['provider']})",
        duration_ms=float(res["latency_ms"]),
        status="ok",
        tags={"provider": res["provider"]},
    )
    return CopilotChatResponse(**res)


@router.get("/ai/config")
async def get_ai_config() -> dict[str, Any]:
    """Return non-secret status of Gemini Decision 1, Gemini Flash, Groq & Ensemble config."""
    return AI_RUNTIME_CONFIG.public_status()


@router.post("/ai/config")
async def update_ai_config(payload: AIConfigUpdate) -> dict[str, Any]:
    """Update runtime Gemini API key, Flash model, Decision 1 toggle & fusion weight."""
    if payload.gemini_api_key is not None:
        AI_RUNTIME_CONFIG.gemini_api_key = payload.gemini_api_key.strip()
    if payload.gemini_flash_model is not None:
        AI_RUNTIME_CONFIG.gemini_flash_model = payload.gemini_flash_model.strip()
    if payload.gemini_decision_enabled is not None:
        AI_RUNTIME_CONFIG.gemini_decision_enabled = payload.gemini_decision_enabled
    if payload.gemini_flash_enabled is not None:
        AI_RUNTIME_CONFIG.gemini_flash_enabled = payload.gemini_flash_enabled
    if payload.decision_fusion_weight is not None:
        AI_RUNTIME_CONFIG.decision_fusion_weight = float(payload.decision_fusion_weight)
    if payload.groq_api_key is not None:
        AI_RUNTIME_CONFIG.groq_api_key = payload.groq_api_key.strip()
    if payload.groq_model is not None:
        AI_RUNTIME_CONFIG.groq_model = payload.groq_model.strip()
    if payload.ensemble_mode is not None and payload.ensemble_mode in ("full", "fast"):
        AI_RUNTIME_CONFIG.ensemble_mode = payload.ensemble_mode

    TELEMETRY_HUB.record_breadcrumb(
        category="ai.config",
        message=f"Updated AI Engine config (Gemini Flash={AI_RUNTIME_CONFIG.gemini_flash_model}, Decision1={AI_RUNTIME_CONFIG.gemini_decision_enabled})",
        level="info",
    )
    return AI_RUNTIME_CONFIG.public_status()


@router.get("/samples", response_model=List[SampleCaseResponse])
async def get_benchmark_samples() -> List[SampleCaseResponse]:
    """List verified clinical benchmark radiograph samples available on the server."""
    samples = [
        SampleCaseResponse(
            id="nih-cardiomegaly",
            name="NIH Case: Cardiomegaly",
            subtitle="Enlarged Cardiac Silhouette (CTR > 0.52)",
            filename="nih_cardiomegaly_00000001.png",
            primary_findings=["Cardiomegaly"],
            suggested_patient=PatientProfile(
                age=68,
                sex="Female",
                setting="Inpatient Cardiology Service",
                indication="Progressive dyspnea on exertion, 3-pillow orthopnea, bilateral pitting ankle edema",
                comorbidities="Congestive Heart Failure (HFrEF, EF 32%), Chronic Kidney Disease",
                smoking="Never smoker",
            ),
        ),
        SampleCaseResponse(
            id="bench-nodule-mass",
            name="Clinical Case: Pulmonary Nodule & Mass",
            subtitle="Discrete Parenchymal Lesion Spectrum",
            filename="benchmark_nodule_mass.jpg",
            primary_findings=["Nodule", "Mass"],
            suggested_patient=PatientProfile(
                age=64,
                sex="Male",
                setting="Pulmonology Outpatient Clinic",
                indication="Chronic dry cough, 14 lb unintentional weight loss over 3 months, mild hemoptysis",
                comorbidities="COPD (GOLD Stage II)",
                smoking="Active smoker (35 pack-years)",
            ),
        ),
        SampleCaseResponse(
            id="nih-mass",
            name="NIH Case: Thoracic Mass & Consolidation",
            subtitle="Focal Space-Occupying Pulmonary Process",
            filename="nih_sample_00027426.png",
            primary_findings=["Mass", "Consolidation"],
            suggested_patient=PatientProfile(
                age=58,
                sex="Female",
                setting="Emergency Department (ED)",
                indication="Subacute right chest discomfort, exertional fatigue, low-grade evening fevers",
                comorbidities="Mild asthma, Osteopenia",
                smoking="Never smoker",
            ),
        ),
        SampleCaseResponse(
            id="bench-atelectasis",
            name="Clinical Case: Atelectasis & Fibrosis",
            subtitle="Subsegmental Volume Loss & Scarring",
            filename="benchmark_pneumonia.jpg",
            primary_findings=["Atelectasis", "Fibrosis"],
            suggested_patient=PatientProfile(
                age=71,
                sex="Male",
                setting="Inpatient Medical Ward",
                indication="Post-operative shallow inspiration, mild hypoxia on room air (SpO2 92%)",
                comorbidities="Interstitial Lung Disease, Hypertension",
                smoking="Former smoker (20 pack-years)",
            ),
        ),
        SampleCaseResponse(
            id="user-pneumonia",
            name="Clinical Case: Acute Pneumonia & Infiltration",
            subtitle="Airspace Infection / Bilateral Check",
            filename="user_xray_pneumonia_infiltrate.jpg",
            primary_findings=["Pneumonia", "Infiltration"],
            suggested_patient=PatientProfile(
                age=72,
                sex="Male",
                setting="Emergency Department (ED)",
                indication="High fever (39.1°C), productive cough with purulent sputum, pleuritic right chest pain",
                comorbidities="Type 2 Diabetes, Hypertension",
                smoking="Former smoker (quit 12y ago)",
            ),
        ),
    ]
    return [s for s in samples if (_BENCHMARK_DIR / s.filename).exists()]


@router.get("/samples/{filename}")
async def get_sample_image(filename: str):
    """Download a raw benchmark radiograph image."""
    file_path = (_BENCHMARK_DIR / filename).resolve()
    try:
        file_path.relative_to(_BENCHMARK_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path)
