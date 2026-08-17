#!/usr/bin/env python3
"""Train DenseNet-121 on NIH ChestX-ray14."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.config import Config  # noqa: E402
from cxr.data.dataset import make_loaders  # noqa: E402
from cxr.seed import set_seed  # noqa: E402
from cxr.training.engine import fit  # noqa: E402
from cxr.training.metrics import format_metrics_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CXR classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg.get("seed", 42)))

    print("=" * 60)
    print("CXR Diagnostic — Training")
    print("=" * 60)
    print(f"Config    : {args.config}")
    print(f"Epochs    : {cfg.get('train.epochs')}")
    print(f"Batch size: {cfg.get('train.batch_size')} × {cfg.get('train.accum_steps')} accum")
    print(f"LR        : {cfg.get('train.lr')}")
    print()

    # Build data loaders.
    loaders = make_loaders(
        manifest_path=cfg.path("data.manifest"),
        image_dir=cfg.path("data.image_dir"),
        image_size=int(cfg.get("data.image_size", 224)),
        batch_size=int(cfg.get("train.batch_size", 16)),
        eval_batch_size=int(cfg.get("eval.batch_size", 32)),
        num_workers=int(cfg.get("data.num_workers", 4)),
        val_fraction=float(cfg.get("data.val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )

    for name, loader in loaders.items():
        print(f"  {name:10s} → {len(loader.dataset):>6d} images, {len(loader):>4d} batches")

    # Train.
    model = fit(cfg, loaders)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
