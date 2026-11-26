"""Persistent User / Patient Clinical Database & Live Evaluation Metrics API endpoints."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import StudyUpdateRequest, UserCreateRequest, UserUpdateRequest
from cxr.db.store import CLINICAL_DB
from cxr.telemetry import TELEMETRY_HUB

router = APIRouter(prefix="/api/v1", tags=["records"])

_EVAL_REPORT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs"
    / "benchmark_results"
    / "full_evaluation_report.json"
)
_LEGACY_EVAL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs"
    / "benchmark_results"
    / "accuracy_comparison_report.json"
)


# ------------------------------------------------------------------
# Users / Patient Clinical Database Endpoints (Non-Blocking Offload)
# ------------------------------------------------------------------


@router.get("/users")
async def list_users() -> list[dict[str, Any]]:
    """List all persisted user / patient records with study counts and latest CTR."""
    return await asyncio.to_thread(CLINICAL_DB.list_users)


@router.post("/users", status_code=201)
async def create_user(payload: UserCreateRequest) -> dict[str, Any]:
    """Create a new persistent user/patient clinical record in SQLite."""
    try:
        user = await asyncio.to_thread(CLINICAL_DB.create_user, payload.model_dump())
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="A user with that MRN already exists."
        ) from exc
    TELEMETRY_HUB.record_breadcrumb(
        category="db.users",
        message=f"Created user record {user['full_name']} ({user['mrn']})",
        level="info",
    )
    return user


@router.get("/users/{user_id}")
async def get_user(user_id: str) -> dict[str, Any]:
    """Retrieve a particular user's full profile, longitudinal CTR history, and diagnostic studies."""
    user = await asyncio.to_thread(CLINICAL_DB.get_user, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return user


@router.patch("/users/{user_id}")
async def update_user(user_id: str, payload: UserUpdateRequest) -> dict[str, Any]:
    """Update a particular user's clinical demographics, history, notes, or UI preferences."""
    try:
        updated = await asyncio.to_thread(
            CLINICAL_DB.update_user, user_id, payload.model_dump(exclude_none=True)
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="MRN conflict with another user record."
        ) from exc
    if not updated:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    TELEMETRY_HUB.record_breadcrumb(
        category="db.users",
        message=f"Updated user profile {updated['full_name']} ({updated['mrn']})",
        level="info",
    )
    return updated


@router.delete("/users/{user_id}")
async def delete_user(user_id: str) -> dict[str, Any]:
    """Delete a user record and cascade-delete associated study records."""
    ok = await asyncio.to_thread(CLINICAL_DB.delete_user, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return {"deleted": True, "user_id": user_id}


# ------------------------------------------------------------------
# Per-User Diagnostic Study & AI Transcript History Endpoints
# ------------------------------------------------------------------


@router.get("/users/{user_id}/studies")
async def list_user_studies(user_id: str, include_payload: bool = False) -> list[dict[str, Any]]:
    """List chronological diagnostic studies & transcripts archived for a specific user."""
    if not await asyncio.to_thread(CLINICAL_DB.get_user, user_id):
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return await asyncio.to_thread(CLINICAL_DB.list_user_studies, user_id, include_payload)


@router.get("/studies/{study_id}")
async def get_study(study_id: str) -> dict[str, Any]:
    """Load a full archived diagnostic study, real AI transcript, and Copilot history."""
    study = await asyncio.to_thread(CLINICAL_DB.get_study, study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    return study


@router.patch("/studies/{study_id}")
async def update_study(study_id: str, payload: StudyUpdateRequest) -> dict[str, Any]:
    """Update an archived study (e.g. edited dictation transcript, re-synthesized report, physician attestation)."""
    updated = await asyncio.to_thread(
        CLINICAL_DB.update_study, study_id, payload.model_dump(exclude_none=True)
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    TELEMETRY_HUB.record_breadcrumb(
        category="db.studies",
        message=f"Updated study {study_id} (signed={bool(updated.get('signed_at'))})",
        level="info",
    )
    return updated


@router.delete("/studies/{study_id}")
async def delete_study(study_id: str) -> dict[str, Any]:
    """Delete a specific archived study record."""
    ok = await asyncio.to_thread(CLINICAL_DB.delete_study, study_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    return {"deleted": True, "study_id": study_id}


@router.get("/studies/{study_id}/copilot")
async def get_study_copilot(study_id: str) -> list[dict[str, Any]]:
    """Retrieve persisted grounded Gemini Copilot consultation transcript for a study."""
    if not await asyncio.to_thread(CLINICAL_DB.get_study, study_id):
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    return await asyncio.to_thread(CLINICAL_DB.list_copilot_messages, study_id)


@router.get("/db/stats")
async def get_db_stats() -> dict[str, Any]:
    """Return SQLite database record statistics."""
    return await asyncio.to_thread(CLINICAL_DB.stats)


# ------------------------------------------------------------------
# Verified ML Evaluation Metrics & Live AI Transcript Benchmarks
# ------------------------------------------------------------------


@router.get("/evaluations")
async def get_evaluation_report() -> dict[str, Any]:
    """Return the latest full ML classification, calibration, biometry, latency & AI transcript evaluation report."""
    for path in (_EVAL_REPORT_PATH, _LEGACY_EVAL_PATH):
        if path.exists():
            try:
                raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8")
                return json.loads(raw_text)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to read evaluation report: {exc}"
                ) from exc
    return {
        "status": "not_run",
        "message": "Run `python scripts/run_full_evaluation.py` to generate live evaluation metrics.",
    }
