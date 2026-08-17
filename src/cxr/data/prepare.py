"""Build manifest.csv from NIH metadata + the resized image folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cxr.constants import PATHOLOGIES
from cxr.data.labels import encode_findings


def _read_id_list(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                ids.add(token)
    return ids


def build_manifest(
    meta_dir: Path,
    image_dir: Path,
    dest: Path,
) -> pd.DataFrame:
    entry_path = meta_dir / "Data_Entry_2017.csv"
    if not entry_path.exists():
        raise FileNotFoundError(
            f"Missing {entry_path}. Run: python scripts/download_nih.py --metadata-only"
        )

    df = pd.read_csv(entry_path)
    # NIH header is messy: "Image Index", "Finding Labels", "Patient ID", ...
    rename = {}
    for col in df.columns:
        low = col.strip().lower()
        if low.startswith("image index"):
            rename[col] = "image_index"
        elif low.startswith("finding"):
            rename[col] = "findings"
        elif low == "patient id":
            rename[col] = "patient_id"
        elif low == "patient age":
            rename[col] = "patient_age"
        elif low == "patient gender":
            rename[col] = "patient_gender"
        elif low == "view position":
            rename[col] = "view_position"
    df = df.rename(columns=rename)

    required = {"image_index", "findings"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data_Entry_2017.csv missing columns {missing}. Got {list(df.columns)}")

    available = {p.stem: p.name for p in image_dir.glob("*.jpg")}
    available.update({p.stem: p.name for p in image_dir.glob("*.png")})
    if not available:
        raise FileNotFoundError(
            f"No images in {image_dir}. Run: python scripts/download_nih.py --max-tars 1"
        )

    df["stem"] = df["image_index"].astype(str).str.replace(".png", "", regex=False)
    df = df[df["stem"].isin(available)].copy()
    df["image_file"] = df["stem"].map(available)
    df["image_id"] = df["image_index"].astype(str)

    labels = np.stack([encode_findings(x) for x in df["findings"].tolist()], axis=0)
    for i, name in enumerate(PATHOLOGIES):
        df[f"y_{name}"] = labels[:, i]

    train_val_path = meta_dir / "train_val_list.txt"
    test_path = meta_dir / "test_list.txt"
    if train_val_path.exists() and test_path.exists():
        train_val = _read_id_list(train_val_path)
        test = _read_id_list(test_path)
        df["split"] = "unknown"
        df.loc[df["image_id"].isin(train_val), "split"] = "train_val"
        df.loc[df["image_id"].isin(test), "split"] = "test"
        # Any leftover images (subset download) go to train_val so we can still train.
        df.loc[df["split"] == "unknown", "split"] = "train_val"
    else:
        # Patient-wise 80/10/10 if official lists are missing.
        if "patient_id" not in df.columns:
            df["patient_id"] = df["image_id"].str.split("_").str[0]
        rng = np.random.default_rng(42)
        patients = df["patient_id"].unique()
        rng.shuffle(patients)
        n = len(patients)
        test_p = set(patients[: int(0.1 * n)])
        val_p = set(patients[int(0.1 * n) : int(0.2 * n)])
        df["split"] = "train"
        df.loc[df["patient_id"].isin(val_p), "split"] = "val"
        df.loc[df["patient_id"].isin(test_p), "split"] = "test"

    dest.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "image_id",
        "image_file",
        "findings",
        "patient_id" if "patient_id" in df.columns else "image_id",
        "split",
        *[f"y_{n}" for n in PATHOLOGIES],
    ]
    # Deduplicate keep list while preserving order.
    seen: set[str] = set()
    ordered = []
    for c in keep:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    out = df[ordered].copy()
    if "patient_id" not in out.columns:
        out["patient_id"] = out["image_id"].str.split("_").str[0]
    out.to_csv(dest, index=False)
    print(f"Wrote {dest}  ({len(out)} images)")
    print(out["split"].value_counts().to_string())
    print("Positive rates:")
    for name in PATHOLOGIES:
        rate = float(out[f"y_{name}"].mean())
        print(f"  {name:22s} {rate:6.3%}  n={int(out[f'y_{name}'].sum())}")
    return out
