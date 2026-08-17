"""PyTorch dataset over the prepared NIH manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from cxr.data.transforms import build_transforms


class ChestXrayDataset(Dataset):
    """One row of manifest.csv = one frontal chest radiograph + 14 labels."""

    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        label_cols = [c for c in self.frame.columns if c.startswith("y_")]
        self.label_cols = label_cols
        self.labels = self.frame[label_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        path = self.image_dir / row["image_file"]
        with Image.open(path) as img:
            image = img.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.from_numpy(self.labels[idx])
        return {
            "image": image,
            "target": target,
            "image_id": str(row["image_id"]),
        }


def _patient_val_split(train_val: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train_val by Patient ID so the same person never appears in both."""
    patients = train_val["patient_id"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val])
    val_mask = train_val["patient_id"].isin(val_patients)
    return train_val.loc[~val_mask].copy(), train_val.loc[val_mask].copy()


def load_splits(
    manifest_path: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(manifest_path)
    if "split" not in df.columns:
        raise ValueError(f"{manifest_path} has no 'split' column. Run prepare first.")

    test = df[df["split"] == "test"].copy()
    train_val = df[df["split"] == "train_val"].copy()
    if train_val.empty:
        # Fallback if we only have a random split column.
        train = df[df["split"] == "train"].copy()
        val = df[df["split"] == "val"].copy()
        return {"train": train, "val": val, "test": test}

    train, val = _patient_val_split(train_val, val_fraction=val_fraction, seed=seed)
    return {"train": train, "val": val, "test": test}


def make_loaders(
    manifest_path: Path,
    image_dir: Path,
    image_size: int = 224,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
    balanced_sampler: bool = True,
) -> dict[str, DataLoader]:
    splits = load_splits(manifest_path, val_fraction=val_fraction, seed=seed)
    loaders: dict[str, DataLoader] = {}
    pin = torch.cuda.is_available()

    for name, frame in splits.items():
        if frame.empty:
            continue
        ds = ChestXrayDataset(
            frame,
            image_dir=image_dir,
            transform=build_transforms(image_size, train=(name == "train")),
        )
        sampler = None
        shuffle = name == "train"
        if name == "train" and balanced_sampler:
            # Weight rare positives higher so Hernia / Pneumonia are seen more often.
            pos_count = ds.labels.sum(axis=1)
            # Images with any finding get higher weight; multi-finding even more.
            weights = np.where(pos_count > 0, 1.0 + pos_count, 0.35).astype(np.float64)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            shuffle = False

        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size if name == "train" else eval_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=(name == "train"),
            persistent_workers=num_workers > 0,
        )
    return loaders
