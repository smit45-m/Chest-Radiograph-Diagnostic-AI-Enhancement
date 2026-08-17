#!/usr/bin/env python3
"""Download NIH ChestX-ray14 images and metadata, then build manifest.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cxr.data.download import download_images, download_metadata  # noqa: E402
from cxr.data.prepare import build_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare NIH ChestX-ray14 dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/nih",
        help="Root directory for dataset files (default: data/nih)",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Resize images to this resolution (default: 224)",
    )
    parser.add_argument(
        "--max-tars", type=int, default=None,
        help="Max image archives to download (default: all 12). Use 1 for a quick start.",
    )
    parser.add_argument(
        "--start-index", type=int, default=1,
        help="1-based index of the first tar to download (default: 1)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Only download CSV/TXT metadata, skip image tars",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_dir = data_dir / "metadata"
    raw_dir = data_dir / "raw"
    image_dir = data_dir / f"images_{args.image_size}"

    # Step 1: Download metadata.
    print("=" * 60)
    print("Step 1 / 3: Downloading metadata")
    print("=" * 60)
    download_metadata(meta_dir)

    if args.metadata_only:
        print("\n--metadata-only set; skipping image download.")
        return

    # Step 2: Download and resize images.
    print("\n" + "=" * 60)
    print("Step 2 / 3: Downloading and resizing images")
    print("=" * 60)
    total = download_images(
        raw_dir=raw_dir,
        image_dir=image_dir,
        image_size=args.image_size,
        max_tars=args.max_tars,
        start_index=args.start_index,
    )
    print(f"\nTotal images ready: {total}")

    # Step 3: Build manifest.
    print("\n" + "=" * 60)
    print("Step 3 / 3: Building manifest.csv")
    print("=" * 60)
    manifest_path = data_dir / "manifest.csv"
    build_manifest(meta_dir=meta_dir, image_dir=image_dir, dest=manifest_path)
    print(f"\nDone. Dataset ready at {data_dir}")


if __name__ == "__main__":
    main()
