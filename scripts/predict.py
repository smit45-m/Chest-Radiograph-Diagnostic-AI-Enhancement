#!/usr/bin/env python3
"""Run single-image inference through the full diagnosis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.config import Config  # noqa: E402
from cxr.serving.pipeline import DiagnosisPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict pathologies from a single chest X-ray"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to chest X-ray image (JPEG/PNG)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.45,
        help="Detection threshold (default: 0.45)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Skip Grad-CAM heatmap generation",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load config (optional).
    try:
        config = Config.load(args.config)
    except Exception:
        config = None

    # Initialize pipeline.
    print("Loading model...")
    with DiagnosisPipeline(
        checkpoint_path=args.checkpoint,
        config=config,
        threshold=args.threshold,
    ) as pipeline:
        # Read image bytes.
        image_bytes = image_path.read_bytes()

        # Run prediction.
        print(f"Predicting: {image_path.name}")
        result = pipeline.predict(image_bytes, threshold=args.threshold)

    # Print results.
    print(f"\nInference time: {result['inference_time_ms']:.1f} ms")
    print(f"\nPositive findings ({len(result['positives'])}):")
    for p in result["positives"]:
        print(f"  {p['name']:22s}  score={p['score']:.3f}  ({p['severity']})")

    if not result["positives"]:
        print("  No pathology detected above threshold.")

    # Save outputs.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Predictions JSON.
    pred_path = out_dir / f"{stem}_predictions.json"
    pred_data = {
        "image": str(image_path),
        "threshold": args.threshold,
        "inference_time_ms": result["inference_time_ms"],
        "predictions": result["predictions"],
        "positives": result["positives"],
    }
    pred_path.write_text(json.dumps(pred_data, indent=2), encoding="utf-8")
    print(f"\nPredictions saved to {pred_path}")

    # Heatmaps.
    if not args.no_heatmap and result["heatmaps"]:
        import base64

        for hm in result["heatmaps"]:
            hm_path = out_dir / f"{stem}_gradcam_{hm['pathology']}.png"
            hm_path.write_bytes(base64.b64decode(hm["image_base64"]))
            print(f"Heatmap saved: {hm_path}")

    # Report.
    if not args.no_report and result.get("report"):
        report = result["report"]
        report_text = report.get("report", report.get("draft", ""))
        report_path = out_dir / f"{stem}_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"Report saved: {report_path}")
        print(f"\n{'─' * 60}")
        print(report_text)
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
