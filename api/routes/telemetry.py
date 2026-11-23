"""Sentry Telemetry, Performance Span & Graphify Knowledge Graph API Routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from cxr.telemetry import TELEMETRY_HUB, build_graphify_payload

router = APIRouter(prefix="/api/v1", tags=["telemetry", "graphify"])


class ClientTelemetryEvent(BaseModel):
    """Client-side Sentry span, breadcrumb, or error event."""

    kind: str = Field(default="span", description="span | breadcrumb | error")
    op: str = Field(default="ui.interaction", description="Operation identifier")
    description: str = Field(default="", description="Event message or span description")
    duration_ms: float = Field(default=0.0, description="Duration in milliseconds for spans")
    level: str = Field(default="info", description="Severity level (info, warning, error)")
    tags: dict[str, Any] = Field(default_factory=dict)


@router.get("/telemetry")
async def get_telemetry_summary() -> dict[str, Any]:
    """Return live Sentry spans, breadcrumbs, error count, and p50/p95/p99 latency stats."""
    return TELEMETRY_HUB.summary()


@router.post("/telemetry/event")
async def ingest_client_telemetry(event: ClientTelemetryEvent) -> dict[str, Any]:
    """Ingest a frontend Sentry span, breadcrumb, or captured error."""
    if event.kind == "error":
        rec = TELEMETRY_HUB.record_error(
            error_type=event.op,
            message=event.description,
            context=event.tags,
        )
        return {"status": "recorded", "event": rec}
    if event.kind == "breadcrumb":
        rec = TELEMETRY_HUB.record_breadcrumb(
            category=event.op,
            message=event.description,
            level=event.level,
            data=event.tags,
        )
        return {"status": "recorded", "event": rec}

    rec = TELEMETRY_HUB.record_span(
        op=event.op,
        description=event.description,
        duration_ms=event.duration_ms,
        status="ok" if event.level != "error" else "error",
        tags=event.tags,
    )
    return {"status": "recorded", "event": rec}


@router.get("/graphify")
async def get_graphify_knowledge_graph() -> dict[str, Any]:
    """Return the Graphify clinical pathology co-occurrence graph and system AST graph."""
    return build_graphify_payload()
