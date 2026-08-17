"""Image transforms. Grayscale CXRs are replicated to 3 channels."""

from __future__ import annotations

from torchvision import transforms

from cxr.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        spatial = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ]
    else:
        spatial = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
        ]

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            *spatial,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def infer_transform(image_size: int = 224) -> transforms.Compose:
    return build_transforms(image_size=image_size, train=False)
