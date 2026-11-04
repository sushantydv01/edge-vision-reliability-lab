"""
Training data loader factory enforcing test-set isolation and training augmentation.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from edge_vision.data.config import load_data_config
from edge_vision.data.dataset import TransformedSubset
from edge_vision.data.splits import get_or_create_split_indices
from edge_vision.data.transforms import get_cifar10_transforms
from edge_vision.training.config import TrainingConfig


def get_training_dataloaders(
    training_config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, torch.Generator]:
    """
    Constructs isolated train and validation DataLoaders for model training.

    Guarantees:
      1. Training-only data augmentation is enabled based on training_config.augmentation.
      2. Validation data transform is strictly unaugmented and deterministic.
      3. The held-out test dataset is never created, referenced, or exposed.
      4. Dedicated torch.Generator is returned to enable precise epoch-boundary resume tracking.

    Parameters:
        training_config: Validated TrainingConfig instance.

    Returns:
        Tuple of (train_loader, val_loader, shuffle_generator).
    """
    data_cfg = load_data_config(training_config.data_config_path)

    # 1. Training transform with explicit augmentation settings
    train_transform = get_cifar10_transforms(
        train=True,
        augment=training_config.augmentation.enabled,
        mean=data_cfg.normalization.mean,
        std=data_cfg.normalization.std,
        random_crop_padding=training_config.augmentation.random_crop_padding,
        random_horizontal_flip=training_config.augmentation.random_horizontal_flip,
    )

    # 2. Validation transform is unaugmented
    eval_transform = get_cifar10_transforms(
        train=False,
        augment=False,
        mean=data_cfg.normalization.mean,
        std=data_cfg.normalization.std,
    )

    # 3. Acquire raw dataset (uses existing raw_dir, no duplicate downloads)
    base_train_dataset = datasets.CIFAR10(
        root=data_cfg.dataset.raw_dir,
        train=True,
        download=data_cfg.dataset.download,
        transform=None,
    )

    # 4. Acquire split indices from the verified manifest
    total_samples = len(base_train_dataset)
    train_idx, val_idx = get_or_create_split_indices(
        manifest_path=data_cfg.split.manifest_path,
        total_samples=total_samples,
        train_size=data_cfg.split.train_size,
        val_size=data_cfg.split.val_size,
        seed=data_cfg.split.seed,
    )

    train_subset = TransformedSubset(base_train_dataset, train_idx, transform=train_transform)
    val_subset = TransformedSubset(base_train_dataset, val_idx, transform=eval_transform)

    # 5. Dedicated generator for reproducible shuffling
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(training_config.seed)

    # 6. DataLoaders
    batch_size = training_config.dataloader.batch_size
    num_workers = training_config.dataloader.num_workers
    pin_memory = training_config.dataloader.pin_memory

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, shuffle_generator
