"""CXR Diagnostic API v4.0 — FastAPI application entry point.

Start with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or via the helper script::

    python scripts/serve.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Ensure ``src/`` is on the import path so ``cxr.*`` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Load project .env automatically so GEMINI_API_KEY / GROQ_API_KEY / CXR_* take effect!
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from cxr.ai.gemini_engine import AI_RUNTIME_CONFIG  # noqa: E402
from cxr.config import Config  # noqa: E402
from cxr.serving.batcher import DynamicBatcher  # noqa: E402
from cxr.serving.pipeline import DiagnosisPipeline  # noqa: E402

from api.routes.health import router as health_router  # noqa: E402
from api.routes.predict import router as predict_router  # noqa: E402
from api.routes.records import router as records_router  # noqa: E402
from api.routes.telemetry import router as telemetry_router  # noqa: E402

AI_RUNTIME_CONFIG.refresh_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load clinical ensemble + start DynamicBatcher on startup; clean up on shutdown."""
    app.state.start_time = time.time()

    config_path = os.environ.get(
        "CXR_CONFIG", str(_PROJECT_ROOT / "configs" / "default.yaml")
    )
    try:
        config = Config.load(config_path)
        print(f"[api] Config loaded from {config_path}")
    except Exception as exc:
        print(f"[api] Could not load config ({exc}); using defaults")
        config = None

    ckpt = os.environ.get("CXR_CHECKPOINT", str(_PROJECT_ROOT / "checkpoints" / "best.pt"))
    threshold = float(config.get("serve.threshold", 0.50)) if config is not None else 0.50
    max_batch = int(config.get("serve.max_batch", 8)) if config is not None else 8
    max_delay_ms = float(config.get("serve.max_delay_ms", 12.0)) if config is not None else 12.0

    try:
        pipeline = DiagnosisPipeline(
            checkpoint_path=ckpt,
            config=config,
            threshold=threshold,
        )
        app.state.pipeline = pipeline

        batcher = DynamicBatcher(
            forward_fn=pipeline.forward_batch,
            max_batch=max_batch,
            max_delay_ms=max_delay_ms,
        )
        batcher.start()
        app.state.batcher = batcher
        print(f"[api] Pipeline & DynamicBatcher ready on {pipeline.device}")
    except Exception as exc:
        print(f"[api] Pipeline init failed: {exc}")
        app.state.pipeline = None
        app.state.batcher = None

    yield

    if getattr(app.state, "batcher", None) is not None:
        await app.state.batcher.stop()
    if getattr(app.state, "pipeline", None) is not None:
        app.state.pipeline.close()
        print("[api] Pipeline closed")


app = FastAPI(
    title="CXR-Diagnostic Pro API",
    description=(
        "Chest Radiograph Multi-Pathology Diagnostic & Report Generator — "
        "Multi-Backbone Transfer Learning Ensemble (DenseNet-121 + ResNet-50 512px + PSPNet CTR), "
        "Grad-CAM++ localization, Gemini Decision 1 consensus, and Gemini Flash report synthesis."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

class _StreamSafeGZipMiddleware(GZipMiddleware):
    """Compress all JSON/static payloads while leaving NDJSON/SSE streaming endpoints unbuffered."""

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("path", "").endswith("/stream"):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_StreamSafeGZipMiddleware, minimum_size=512)


@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    elif path.startswith("/api/v1/samples/"):
        response.headers.setdefault("Cache-Control", "public, max-age=3600")
    return response

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(records_router)
app.include_router(telemetry_router)

# Serve compiled React/Vite frontend (web/assets + web/index.html)
_WEB_DIR = _PROJECT_ROOT / "web"
if (_WEB_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_WEB_DIR / "assets")), name="assets")
if _WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive PACS Diagnostic Studio SPA at root /."""
    index_file = _WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")


@app.get("/{filename:path}", include_in_schema=False)
async def serve_static_root(filename: str):
    """Serve relative static files when requested from root /."""
    if filename.startswith("api/") or filename in ("docs", "redoc", "openapi.json"):
        return RedirectResponse(url="/docs")
    target = (_WEB_DIR / filename).resolve()
    try:
        target.relative_to(_WEB_DIR.resolve())
    except ValueError:
        return RedirectResponse(url="/")
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    index_file = _WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/")
