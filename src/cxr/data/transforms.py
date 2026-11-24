"""Radiology-safe image transforms. Grayscale CXRs are replicated to 3 channels.

Note: Horizontal flipping is DISABLED by default (`allow_hflip=False`) because thoracic
anatomy is chiral (heart & aortic arch on the left, liver on the right); horizontal flips
degrade Cardiomegaly, Hernia, and mediastinal calibration.
"""

from __future__ import annotations

from torchvision import transforms

from cxr.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    allow_hflip: bool = False,
) -> transforms.Compose:
    if train:
        spatial: list = [
            transforms.Resize(int(image_size * 1.12)),
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(degrees=6),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.10, contrast=0.12),
        ]
        if allow_hflip:
            spatial.insert(2, transforms.RandomHorizontalFlip(p=0.5))
    else:
        spatial = [
            transforms.Resize(int(image_size * 1.12)),
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
