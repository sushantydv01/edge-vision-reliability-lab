"""
Unit tests for CIFAR-10 data pipeline, transforms, deterministic splitting, and compact manifest management.
Uses lightweight synthetic datasets to avoid network dependency during unit tests.
"""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from edge_vision.data import (
    CIFAR10_CLASSES,
    TransformedSubset,
    compute_indices_checksum,
    generate_train_val_split_indices,
    get_cifar10_transforms,
    get_or_create_split_indices,
    load_data_config,
    load_split_manifest,
    save_split_manifest,
    validate_manifest_schema,
    validate_split_indices,
)


class DummyCIFAR10(Dataset):
    """Synthetic dataset mirroring torchvision CIFAR10 internal attributes."""

    def __init__(self, size: int = 100):
        rng = np.random.default_rng(42)
        self.data = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = [int(x) for x in rng.integers(0, 10, size=size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_cifar10_class_mapping():
    assert len(CIFAR10_CLASSES) == 10
    expected = (
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
    assert CIFAR10_CLASSES == expected


def test_deterministic_split_properties():
    total = 50000
    train_size = 45000
    val_size = 5000

    train_idx_1, val_idx_1 = generate_train_val_split_indices(
        total_samples=total, train_size=train_size, val_size=val_size, seed=42
    )
    train_idx_2, val_idx_2 = generate_train_val_split_indices(
        total_samples=total, train_size=train_size, val_size=val_size, seed=42
    )

    # 1. Determinism
    assert train_idx_1 == train_idx_2
    assert val_idx_1 == val_idx_2

    # 2. Lengths
    assert len(train_idx_1) == 45000
    assert len(val_idx_1) == 5000

    # 3. Disjointness
    set_train = set(train_idx_1)
    set_val = set(val_idx_1)
    assert len(set_train.intersection(set_val)) == 0

    # 4. Full accounting of all original indices
    union_set = set_train.union(set_val)
    assert len(union_set) == total
    assert union_set == set(range(total))

    # 5. Seed sensitivity
    train_diff_seed, _ = generate_train_val_split_indices(
        total_samples=total, train_size=train_size, val_size=val_size, seed=999
    )
    assert train_idx_1 != train_diff_seed


def test_validate_split_indices_failures():
    # 1. Overlapping splits (total count matches, but elements overlap)
    with pytest.raises(ValueError, match="overlap"):
        validate_split_indices([0, 1, 2], [2, 3], total_samples=5)

    # 2. Incomplete coverage (count mismatch)
    with pytest.raises(ValueError, match="do not match total expected samples"):
        validate_split_indices([0, 1], [2, 3], total_samples=5)

    # 3. Duplicate indices
    with pytest.raises(ValueError, match="duplicate"):
        validate_split_indices([0, 0, 1], [2, 3], total_samples=5)

    # 4. Invalid types
    with pytest.raises(TypeError, match="lists or tuples"):
        validate_split_indices("invalid", [0, 1], total_samples=5)


def test_compact_manifest_creation_and_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "compact_split.json"
        train_idx = [0, 2, 4]
        val_idx = [1, 3]

        payload = save_split_manifest(train_idx, val_idx, manifest_path=str(manifest_file), seed=42)
        assert manifest_file.is_file()
        assert manifest_file.stat().st_size < 2000  # Must be compact (<2 KB)

        data = load_split_manifest(str(manifest_file))
        assert data["seed"] == 42
        assert data["train_size"] == 3
        assert data["val_size"] == 2
        assert data["train_indices_checksum"] == compute_indices_checksum(train_idx)
        assert data["val_indices_checksum"] == compute_indices_checksum(val_idx)


def test_atomic_manifest_writing():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "atomic_manifest.json"
        train_idx = [0, 2, 4]
        val_idx = [1, 3]

        save_split_manifest(train_idx, val_idx, manifest_path=str(manifest_file), seed=42)
        assert manifest_file.is_file()

        # Check that no temporary files remain in the directory
        temp_files = list(Path(tmpdir).glob("*.tmp.*"))
        assert len(temp_files) == 0

        # Verify content integrity
        data = load_split_manifest(str(manifest_file))
        assert data["seed"] == 42


def test_stale_manifest_ignored_when_seed_changes():
    """Regression test: requesting seed 999 must not return seed 42 split from existing manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "manifest.json"

        # Generate manifest with seed 42
        train_42, val_42 = get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=90,
            val_size=10,
            seed=42,
        )

        # Request with seed 999 pointing to same manifest path
        train_999, val_999 = get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=90,
            val_size=10,
            seed=999,
        )

        # Must NOT match seed 42 indices
        assert train_999 != train_42
        assert val_999 != val_42

        # Must match direct generation for seed 999
        expected_train_999, expected_val_999 = generate_train_val_split_indices(100, 90, 10, seed=999)
        assert train_999 == expected_train_999
        assert val_999 == expected_val_999


def test_configuration_mismatch_sizes_regenerates_indices():
    """Verify that when split sizes differ from manifest, indices are regenerated for requested configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "manifest.json"

        # Manifest has 80/20 split
        get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=80,
            val_size=20,
            seed=42,
        )

        # Request 90/10 split with same seed
        train_idx, val_idx = get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=90,
            val_size=10,
            seed=42,
        )
        assert len(train_idx) == 90
        assert len(val_idx) == 10


def test_manifest_checksum_mismatch_detection_and_regeneration():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "manifest.json"

        # Create valid manifest
        get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=90,
            val_size=10,
            seed=42,
        )

        # Corrupt the train checksum in manifest (valid 64-char hex but mismatched)
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)
        manifest_data["train_indices_checksum"] = "0" * 64
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

        # Must raise RuntimeError on checksum mismatch when strict
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            get_or_create_split_indices(
                manifest_path=str(manifest_file),
                total_samples=100,
                train_size=90,
                val_size=10,
                seed=42,
                verify_checksum=True,
            )

        # Force regenerate should repair it
        train_repaired, val_repaired = get_or_create_split_indices(
            manifest_path=str(manifest_file),
            total_samples=100,
            train_size=90,
            val_size=10,
            seed=42,
            force_regenerate=True,
        )
        assert len(train_repaired) == 90
        assert len(val_repaired) == 10


def test_malformed_manifest_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_file = Path(tmpdir) / "malformed.json"

        # 1. Invalid JSON
        with open(manifest_file, "w") as f:
            f.write("{ invalid json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_split_manifest(str(manifest_file))

        # 2. Missing required keys
        with open(manifest_file, "w") as f:
            json.dump({"dataset": "cifar10"}, f)
        with pytest.raises(ValueError, match="missing required field"):
            load_split_manifest(str(manifest_file))

        # 3. Invalid boolean for integer field
        invalid_data = {
            "dataset": "cifar10",
            "total_samples": 50000,
            "train_size": True,
            "val_size": 5000,
            "seed": 42,
            "algorithm": "torch.randperm_v1",
            "pytorch_version": "2.15.0",
            "train_indices_checksum": "a" * 64,
            "val_indices_checksum": "b" * 64,
        }
        with open(manifest_file, "w") as f:
            json.dump(invalid_data, f)
        with pytest.raises(ValueError, match="must be an integer, not a boolean"):
            load_split_manifest(str(manifest_file))

        # 4. Invalid checksum formatting (too short)
        invalid_data["train_size"] = 45000
        invalid_data["train_indices_checksum"] = "short_hash"
        with open(manifest_file, "w") as f:
            json.dump(invalid_data, f)
        with pytest.raises(ValueError, match="valid 64-character lowercase hex"):
            load_split_manifest(str(manifest_file))

        # 5. Unsupported algorithm
        invalid_data["train_indices_checksum"] = "a" * 64
        invalid_data["algorithm"] = "unsupported_v99"
        with open(manifest_file, "w") as f:
            json.dump(invalid_data, f)
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            load_split_manifest(str(manifest_file))


def test_subsets_share_underlying_dataset_storage():
    """Verify that both train and validation subsets share the exact same underlying base_dataset object."""
    dummy = DummyCIFAR10(size=50)
    transform = get_cifar10_transforms(train=False, augment=False)
    train_subset = TransformedSubset(dummy, list(range(40)), transform=transform)
    val_subset = TransformedSubset(dummy, list(range(40, 50)), transform=transform)

    # Identical object reference in memory (no dataset cloning)
    assert train_subset.base_dataset is val_subset.base_dataset
    assert train_subset.base_dataset is dummy

    # Mutation on base reflects directly in subsets without copying
    original_val = dummy.data[42].copy()
    dummy.data[42, 0, 0, 0] = 255
    assert np.array_equal(val_subset.base_dataset.data[42], dummy.data[42])
    # Restore
    dummy.data[42] = original_val


def test_exact_normalization_values_known_input():
    """Verify normalization formula (x - mean) / std against canonical CIFAR-10 constants."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform = get_cifar10_transforms(train=False, augment=False, mean=mean, std=std)

    # All-black image (0 uint8 -> 0.0 float)
    black_img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    black_tensor = transform(black_img)

    expected_black_r = (0.0 - mean[0]) / std[0]  # -1.98947...
    expected_black_g = (0.0 - mean[1]) / std[1]  # -1.98028...
    expected_black_b = (0.0 - mean[2]) / std[2]  # -1.70680...

    assert torch.allclose(black_tensor[0, :, :], torch.tensor(expected_black_r), atol=1e-4)
    assert torch.allclose(black_tensor[1, :, :], torch.tensor(expected_black_g), atol=1e-4)
    assert torch.allclose(black_tensor[2, :, :], torch.tensor(expected_black_b), atol=1e-4)

    # All-white image (255 uint8 -> 1.0 float)
    white_img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    white_tensor = transform(white_img)

    expected_white_r = (1.0 - mean[0]) / std[0]  # 2.05910...
    expected_white_g = (1.0 - mean[1]) / std[1]  # 2.12648...
    expected_white_b = (1.0 - mean[2]) / std[2]  # 2.11582...

    assert torch.allclose(white_tensor[0, :, :], torch.tensor(expected_white_r), atol=1e-4)
    assert torch.allclose(white_tensor[1, :, :], torch.tensor(expected_white_g), atol=1e-4)
    assert torch.allclose(white_tensor[2, :, :], torch.tensor(expected_white_b), atol=1e-4)


def test_eval_transform_determinism():
    transform = get_cifar10_transforms(train=False, augment=False)
    raw_img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))

    t1 = transform(raw_img)
    t2 = transform(raw_img)

    assert isinstance(t1, torch.Tensor)
    assert t1.shape == (3, 32, 32)
    assert t1.dtype == torch.float32
    # Evaluation transform must be completely deterministic
    assert torch.all(t1 == t2)


def test_augmentation_produces_stochastic_variation():
    """Verify that training augmentation produces actual random variations on a non-symmetric image."""
    transform = get_cifar10_transforms(train=True, augment=True)

    # Create a non-symmetric image with high directional contrast (diagonal gradient)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        for j in range(32):
            arr[i, j] = [i * 8, j * 8, (i + j) * 4]
    img = Image.fromarray(arr)

    # Sample multiple transformed outputs
    samples = [transform(img) for _ in range(30)]

    # At least two samples must differ due to random cropping and horizontal flipping
    differences = [not torch.allclose(samples[0], s) for s in samples[1:]]
    assert any(differences), "Augmentation pipeline failed to produce stochastic variations on non-symmetric input."


def test_dataloader_batch_dimensions_and_labels():
    dummy = DummyCIFAR10(size=32)
    transform = get_cifar10_transforms(train=False, augment=False)
    subset = TransformedSubset(dummy, list(range(32)), transform=transform)

    batch_size = 8
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    batches = list(loader)
    assert len(batches) == 4

    images, labels = batches[0]
    assert images.shape == (batch_size, 3, 32, 32)
    assert labels.shape == (batch_size,)
    assert labels.min().item() >= 0
    assert labels.max().item() <= 9


def test_dataloader_reproducible_shuffling():
    dummy = DummyCIFAR10(size=64)
    transform = get_cifar10_transforms(train=False, augment=False)
    subset = TransformedSubset(dummy, list(range(64)), transform=transform)

    gen1 = torch.Generator().manual_seed(123)
    loader1 = DataLoader(subset, batch_size=16, shuffle=True, generator=gen1)
    labels1 = [labels.tolist() for _, labels in loader1]

    gen2 = torch.Generator().manual_seed(123)
    loader2 = DataLoader(subset, batch_size=16, shuffle=True, generator=gen2)
    labels2 = [labels.tolist() for _, labels in loader2]

    assert labels1 == labels2


def test_evaluation_dataloader_ordering_is_stable():
    """Verify that evaluation DataLoader (shuffle=False) yields identical batch order across multiple passes."""
    dummy = DummyCIFAR10(size=40)
    transform = get_cifar10_transforms(train=False, augment=False)
    subset = TransformedSubset(dummy, list(range(40)), transform=transform)

    eval_loader = DataLoader(subset, batch_size=10, shuffle=False, num_workers=0)

    pass1_labels = [labels.tolist() for _, labels in eval_loader]
    pass2_labels = [labels.tolist() for _, labels in eval_loader]

    assert pass1_labels == pass2_labels
    assert [item for sublist in pass1_labels for item in sublist] == [dummy.targets[i] for i in range(40)]
