#!/usr/bin/env python3
"""Evaluate a trained CXR model on the test (or val) split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.config import Config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.models.densenet import DenseNet121Classifier  # noqa: E402
from cxr.seed import set_seed  # noqa: E402
from cxr.training.engine import evaluate  # noqa: E402
from cxr.training.metrics import format_metrics_table, save_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CXR model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("CXR Diagnostic — Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Threshold : {args.threshold}")
    print(f"Device    : {device}")
    print()

    # Load model.
    model = DenseNet121Classifier.load(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Build data loaders.
    loaders = make_loaders(
        manifest_path=cfg.path("data.manifest"),
        image_dir=cfg.path("data.image_dir"),
        image_size=int(cfg.get("data.image_size", 224)),
        batch_size=int(cfg.get("eval.batch_size", 32)),
        eval_batch_size=int(cfg.get("eval.batch_size", 32)),
        num_workers=int(cfg.get("data.num_workers", 4)),
        val_fraction=float(cfg.get("data.val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
        balanced_sampler=False,
    )

    loader = loaders.get(args.split)
    if loader is None:
        print(f"Split '{args.split}' not found. Available: {list(loaders.keys())}")
        sys.exit(1)

    print(f"Evaluating on {len(loader.dataset)} images ({len(loader)} batches)\n")

    # Run evaluation.
    metrics = evaluate(model, loader, device, threshold=args.threshold)

    # Print results.
    print(format_metrics_table(metrics))

    # Save metrics to JSON.
    out_dir = cfg.path("output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{args.split}_metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
