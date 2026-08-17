#!/usr/bin/env python3
"""Start the CXR Diagnostic FastAPI server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the CXR Diagnostic API server")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload (dev)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    args = parser.parse_args()

    # Set config env var so api.main picks it up.
    os.environ["CXR_CONFIG"] = str(Path(args.config).resolve())

    # Read host/port from config if not overridden.
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    try:
        from cxr.config import Config
        cfg = Config.load(args.config)
        if args.host is None:
            host = cfg.get("serve.host", "0.0.0.0")
        if args.port is None:
            port = int(cfg.get("serve.port", 8000))
    except Exception:
        pass

    print(f"Starting CXR Diagnostic API at http://{host}:{port}")
    print(f"Config: {args.config}")
    print(f"Swagger docs: http://{host}:{port}/docs")
    print(f"PACS Web UI: http://{host}:{port}/")
    print()

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=args.workers,
        app_dir=str(_ROOT),
    )


if __name__ == "__main__":
    main()
