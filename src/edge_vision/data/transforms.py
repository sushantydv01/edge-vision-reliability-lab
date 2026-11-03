"""
Image transformations for CIFAR-10 data pipeline.
"""

from typing import Sequence
import torchvision.transforms as transforms


def get_cifar10_transforms(
    train: bool = False,
    augment: bool = False,
    mean: Sequence[float] = (0.4914, 0.4822, 0.4465),
    std: Sequence[float] = (0.2470, 0.2435, 0.2616),
    random_crop_padding: int = 4,
    random_horizontal_flip: bool = True,
) -> transforms.Compose:
    """
    Construct torchvision transforms for CIFAR-10.

    Args:
        train: Whether the transform is for training split.
        augment: Whether to apply data augmentation (only applies if train=True).
        mean: Channel-wise mean values for normalization.
        std: Channel-wise std values for normalization.
        random_crop_padding: Padding size if random crop is enabled.
        random_horizontal_flip: Whether to apply random horizontal flip.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []

    if train and augment:
        if random_crop_padding > 0:
            transform_list.append(transforms.RandomCrop(32, padding=random_crop_padding))
        if random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

    # Canonical tensor conversion and normalization
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
