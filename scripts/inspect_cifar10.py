#!/usr/bin/env python3
"""
CIFAR-10 dataset acquisition, integrity verification, and visual sample generator.
Downloads CIFAR-10 into data/raw/ (if not present), verifies dataset integrity,
and saves a sample grid figure into reports/figures/cifar10_samples.png.
"""

from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets

from edge_vision.data import (
    CIFAR10_CLASSES,
    get_cifar10_dataloaders,
    load_data_config,
)


def verify_dataset_integrity(raw_dir: str = "data/raw"):
    print("=" * 60)
    print("CIFAR-10 Dataset Acquisition & Integrity Inspection")
    print("=" * 60)

    # 1. Download / Load raw datasets
    print(f"Acquiring raw CIFAR-10 dataset into: {raw_dir} ...")
    raw_train = datasets.CIFAR10(root=raw_dir, train=True, download=True)
    raw_test = datasets.CIFAR10(root=raw_dir, train=False, download=True)

    # 2. Check counts
    train_count = len(raw_train)
    test_count = len(raw_test)
    print(f"[-] Training set sample count: {train_count} (Expected: 50,000)")
    print(f"[-] Test set sample count:     {test_count} (Expected: 10,000)")
    assert train_count == 50000, f"Expected 50000 train samples, got {train_count}"
    assert test_count == 10000, f"Expected 10000 test samples, got {test_count}"

    # 3. Check image format & dimensions
    train_data = raw_train.data
    test_data = raw_test.data
    print(f"[-] Raw train data shape:      {train_data.shape} (N, H, W, C)")
    print(f"[-] Raw train data dtype:      {train_data.dtype}")
    assert train_data.shape == (50000, 32, 32, 3), f"Unexpected train data shape: {train_data.shape}"
    assert test_data.shape == (10000, 32, 32, 3), f"Unexpected test data shape: {test_data.shape}"

    # 4. Check classes and labels
    train_targets = raw_train.targets
    unique_labels = sorted(list(set(train_targets)))
    print(f"[-] Unique labels in training: {unique_labels}")
    assert unique_labels == list(range(10)), f"Unexpected labels: {unique_labels}"
    assert len(CIFAR10_CLASSES) == 10, "Expected 10 class names"
    print(f"[-] Class names: {CIFAR10_CLASSES}")

    # 5. Class balance
    counter = Counter(train_targets)
    print(f"[-] Class counts in training:  {dict(counter)}")
    for cls_idx in range(10):
        assert counter[cls_idx] == 5000, f"Class {cls_idx} has {counter[cls_idx]} samples (expected 5000)"

    print("[-] Dataset integrity checks: ALL PASSED\n")
    return raw_train, raw_test


def generate_sample_figure(raw_train, output_path: str = "reports/figures/cifar10_samples.png"):
    """Generate a clean visual sample grid showing one representative sample per class."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find first sample for each class
    class_samples = {}
    for img, label in zip(raw_train.data, raw_train.targets):
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == 10:
            break

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    fig.suptitle("CIFAR-10 Dataset - Class Samples (32x32 RGB)", fontsize=13, fontweight="bold")

    for label_idx in range(10):
        ax = axes[label_idx // 5, label_idx % 5]
        img = class_samples[label_idx]
        ax.imshow(img)
        ax.set_title(f"{label_idx}: {CIFAR10_CLASSES[label_idx]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[-] Saved sample grid to: {output_path}")


def inspect_dataloaders(config_path: str = "configs/cifar10.yaml"):
    print("\n" + "=" * 60)
    print("DataLoader & Split Inspection")
    print("=" * 60)

    cfg = load_data_config(config_path)
    print(f"[-] Configuration loaded from: {config_path}")
    print(f"    - Split Seed:       {cfg.split.seed}")
    print(f"    - Train split size: {cfg.split.train_size}")
    print(f"    - Val split size:   {cfg.split.val_size}")
    print(f"    - Batch size:       {cfg.dataloader.batch_size}")
    print(f"    - Worker count:     {cfg.dataloader.num_workers}")
    print(f"    - Normalization:    mean={cfg.normalization.mean}, std={cfg.normalization.std}")

    loaders = get_cifar10_dataloaders(cfg)

    train_batches = len(loaders.train)
    val_batches = len(loaders.val)
    test_batches = len(loaders.test)

    print(f"[-] Train DataLoader: {len(loaders.train.dataset)} samples across {train_batches} batches")
    print(f"[-] Val DataLoader:   {len(loaders.val.dataset)} samples across {val_batches} batches")
    print(f"[-] Test DataLoader:  {len(loaders.test.dataset)} samples across {test_batches} batches")

    # Fetch one batch from each loader to verify tensor shapes and ranges
    for name, loader in [("Train", loaders.train), ("Val", loaders.val), ("Test", loaders.test)]:
        images, targets = next(iter(loader))
        print(f"[-] {name} first batch:")
        print(f"    - Tensor shape: {images.shape} (B, C, H, W)")
        print(f"    - Tensor dtype: {images.dtype}")
        print(f"    - Label shape:  {targets.shape}")
        print(f"    - Label range:  min={targets.min().item()}, max={targets.max().item()}")
        print(f"    - Value range:  min={images.min().item():.3f}, max={images.max().item():.3f}")
        assert images.shape == (cfg.dataloader.batch_size, 3, 32, 32)
        assert targets.shape == (cfg.dataloader.batch_size,)

    print("\n[-] DataLoader inspection: ALL PASSED")


def main():
    raw_train, raw_test = verify_dataset_integrity(raw_dir="data/raw")
    generate_sample_figure(raw_train, output_path="reports/figures/cifar10_samples.png")
    inspect_dataloaders(config_path="configs/cifar10.yaml")
    print("\n" + "=" * 60)
    print("CIFAR-10 Pipeline Inspection Complete: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
