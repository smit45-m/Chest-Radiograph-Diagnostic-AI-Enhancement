"""CXR Diagnostic API — FastAPI application entry point.

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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Ensure ``src/`` is on the import path so ``cxr.*`` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from cxr.config import Config  # noqa: E402
from cxr.serving.pipeline import DiagnosisPipeline  # noqa: E402

from api.routes.health import router as health_router  # noqa: E402
from api.routes.predict import router as predict_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    app.state.start_time = time.time()

    # Load configuration.
    config_path = os.environ.get(
        "CXR_CONFIG", str(_PROJECT_ROOT / "configs" / "default.yaml")
    )
    try:
        config = Config.load(config_path)
        print(f"[api] Config loaded from {config_path}")
    except Exception as exc:
        print(f"[api] Could not load config ({exc}); using defaults")
        config = None

    # Initialise the inference pipeline.
    ckpt = os.environ.get("CXR_CHECKPOINT", str(_PROJECT_ROOT / "checkpoints" / "best.pt"))
    threshold = 0.45
    if config is not None:
        threshold = float(config.get("serve.threshold", 0.45))

    try:
        pipeline = DiagnosisPipeline(
            checkpoint_path=ckpt,
            config=config,
            threshold=threshold,
        )
        app.state.pipeline = pipeline
        print(f"[api] Pipeline ready on {pipeline.device}")
    except Exception as exc:
        print(f"[api] Pipeline init failed: {exc}")
        app.state.pipeline = None

    yield

    # Shutdown.
    if getattr(app.state, "pipeline", None) is not None:
        app.state.pipeline.close()
        print("[api] Pipeline closed")


app = FastAPI(
    title="CXR Diagnostic API",
    description=(
        "Chest Radiograph Multi-Pathology Diagnostic & Report Generator — "
        "DenseNet-121 classifier with Grad-CAM and LLM report synthesis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)

# Mount web frontend static assets
_WEB_DIR = _PROJECT_ROOT / "web"
if _WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")

@app.get("/", include_in_schema=False)
async def root():
    """Serve the interactive PACS Diagnostic frontend at root /"""
    index_file = _WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")

@app.get("/{filename:path}", include_in_schema=False)
async def serve_static_root(filename: str):
    """Serve relative static files (e.g. styles.css, app.js) when requested from root /"""
    target = _WEB_DIR / filename
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    return RedirectResponse(url="/")
