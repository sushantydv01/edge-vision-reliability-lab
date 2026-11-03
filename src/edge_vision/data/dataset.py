"""
CIFAR-10 dataset loading, subset creation, and DataLoader construction.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from edge_vision.data.config import DataPipelineConfig, load_data_config
from edge_vision.data.splits import get_or_create_split_indices
from edge_vision.data.transforms import get_cifar10_transforms

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class TransformedSubset(Dataset):
    """
    Lightweight indexed view with on-the-fly transforms over a base CIFAR-10 dataset.
    Shares the underlying image array in host memory without duplication.
    """

    def __init__(self, base_dataset: datasets.CIFAR10, indices: Sequence[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img = self.base_dataset.data[real_idx]
        target = int(self.base_dataset.targets[real_idx])

        # Convert numpy array (H, W, C) to PIL Image for torchvision transform compatibility
        pil_img = Image.fromarray(img)
        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.functional.to_tensor(pil_img)

        return tensor_img, target


@dataclass
class CIFAR10DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_indices: Sequence[int]
    val_indices: Sequence[int]


def get_cifar10_dataloaders(
    config: Optional[DataPipelineConfig] = None,
    config_path: Optional[str] = None,
) -> CIFAR10DataLoaders:
    """
    Construct memory-conscious, reproducible CIFAR-10 DataLoaders for train, val, and test splits.

    Args:
        config: Populated DataPipelineConfig (if None, loaded from config_path).
        config_path: Path to YAML config file (defaults to configs/cifar10.yaml).

    Returns:
        CIFAR10DataLoaders containing (train, val, test) DataLoaders and split indices.
    """
    if config is None:
        config = load_data_config(config_path)

    # 1. Transforms
    train_transform = get_cifar10_transforms(
        train=True,
        augment=config.augmentation.enabled,
        mean=config.normalization.mean,
        std=config.normalization.std,
        random_crop_padding=config.augmentation.random_crop_padding,
        random_horizontal_flip=config.augmentation.random_horizontal_flip,
    )

    eval_transform = get_cifar10_transforms(
        train=False,
        augment=False,
        mean=config.normalization.mean,
        std=config.normalization.std,
    )

    # 2. Acquire raw datasets (torchvision handles caching in raw_dir)
    base_train_dataset = datasets.CIFAR10(
        root=config.dataset.raw_dir,
        train=True,
        download=config.dataset.download,
        transform=None,
    )

    test_dataset = datasets.CIFAR10(
        root=config.dataset.raw_dir,
        train=False,
        download=config.dataset.download,
        transform=eval_transform,
    )

    # 3. Deterministic train / validation split
    total_samples = len(base_train_dataset)
    train_idx, val_idx = get_or_create_split_indices(
        manifest_path=config.split.manifest_path,
        total_samples=total_samples,
        train_size=config.split.train_size,
        val_size=config.split.val_size,
        seed=config.split.seed,
    )

    train_subset = TransformedSubset(base_train_dataset, train_idx, transform=train_transform)
    val_subset = TransformedSubset(base_train_dataset, val_idx, transform=eval_transform)

    # 4. Deterministic generator for shuffling
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(config.split.seed)

    # 5. DataLoaders (memory-conservative defaults)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle_train,
        generator=shuffle_generator if config.dataloader.shuffle_train else None,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=False,
    )

    return CIFAR10DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_indices=train_idx,
        val_indices=val_idx,
    )
