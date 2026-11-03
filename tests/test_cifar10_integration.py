"""
Integration tests for CIFAR-10 data pipeline using local downloaded data.
Verifies full dataset integrity, split sizes, DataLoader batch shapes,
test set isolation, and compact manifest checksum matching.
"""

from pathlib import Path
import pytest
import torch
import torchvision.datasets as datasets

from edge_vision.data import (
    CIFAR10_CLASSES,
    compute_indices_checksum,
    get_cifar10_dataloaders,
    get_or_create_split_indices,
    load_data_config,
    load_split_manifest,
)


@pytest.mark.skipif(
    not (Path("data/raw/cifar-10-batches-py").exists() or Path("data/raw/cifar-10-python.tar.gz").exists()),
    reason="CIFAR-10 raw dataset is not downloaded.",
)
def test_cifar10_raw_dataset_integrity():
    raw_train = datasets.CIFAR10(root="data/raw", train=True, download=False)
    raw_test = datasets.CIFAR10(root="data/raw", train=False, download=False)

    assert len(raw_train) == 50000
    assert len(raw_test) == 10000
    assert raw_train.data.shape == (50000, 32, 32, 3)
    assert raw_test.data.shape == (10000, 32, 32, 3)
    assert raw_train.data.dtype.name == "uint8"
    assert raw_test.data.dtype.name == "uint8"
    assert len(set(raw_train.targets)) == 10
    assert len(set(raw_test.targets)) == 10


@pytest.mark.skipif(
    not (Path("data/raw/cifar-10-batches-py").exists() or Path("data/raw/cifar-10-python.tar.gz").exists()),
    reason="CIFAR-10 raw dataset is not downloaded.",
)
def test_cifar10_dataloaders_end_to_end():
    config = load_data_config("configs/cifar10.yaml")
    loaders = get_cifar10_dataloaders(config)

    # Verify split sample counts
    assert len(loaders.train.dataset) == 45000
    assert len(loaders.val.dataset) == 5000
    assert len(loaders.test.dataset) == 10000

    # Verify batch iteration on train loader
    images, labels = next(iter(loaders.train))
    assert images.shape == (config.dataloader.batch_size, 3, 32, 32)
    assert images.dtype == torch.float32
    assert labels.shape == (config.dataloader.batch_size,)
    assert labels.min().item() >= 0
    assert labels.max().item() <= 9

    # Verify batch iteration on val loader
    val_images, val_labels = next(iter(loaders.val))
    assert val_images.shape == (config.dataloader.batch_size, 3, 32, 32)
    assert val_images.dtype == torch.float32
    assert val_labels.shape == (config.dataloader.batch_size,)

    # Verify batch iteration on test loader
    test_images, test_labels = next(iter(loaders.test))
    assert test_images.shape == (config.dataloader.batch_size, 3, 32, 32)
    assert test_images.dtype == torch.float32
    assert test_labels.shape == (config.dataloader.batch_size,)


@pytest.mark.skipif(
    not (Path("data/raw/cifar-10-batches-py").exists() or Path("data/raw/cifar-10-python.tar.gz").exists()),
    reason="CIFAR-10 raw dataset is not downloaded.",
)
def test_cifar10_test_set_isolation():
    """Verify official 10,000-sample test set is completely separate from train/validation partitions."""
    config = load_data_config("configs/cifar10.yaml")
    loaders = get_cifar10_dataloaders(config)

    train_indices = set(loaders.train_indices)
    val_indices = set(loaders.val_indices)

    # Check partition sizes and coverage on the 50,000 training set
    assert len(train_indices) == 45000
    assert len(val_indices) == 5000
    assert train_indices.union(val_indices) == set(range(50000))
    assert len(train_indices.intersection(val_indices)) == 0

    # Test dataset is separate instance of len 10000
    assert len(loaders.test.dataset) == 10000
    assert loaders.test.dataset is not loaders.train.dataset
    assert loaders.test.dataset is not loaders.val.dataset


@pytest.mark.skipif(
    not Path("configs/cifar10_split_manifest.json").exists(),
    reason="Manifest file does not exist.",
)
def test_cifar10_manifest_checksum_integrity():
    manifest = load_split_manifest("configs/cifar10_split_manifest.json")
    train_idx, val_idx = get_or_create_split_indices(
        manifest_path="configs/cifar10_split_manifest.json",
        total_samples=50000,
        train_size=45000,
        val_size=5000,
        seed=42,
        verify_checksum=True,
    )

    actual_train_cksum = compute_indices_checksum(train_idx)
    actual_val_cksum = compute_indices_checksum(val_idx)

    assert actual_train_cksum == manifest["train_indices_checksum"]
    assert actual_val_cksum == manifest["val_indices_checksum"]
    assert list(train_idx[:5]) == manifest["sample_verification"]["first_5_train_indices"]
    assert list(val_idx[:5]) == manifest["sample_verification"]["first_5_val_indices"]
