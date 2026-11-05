"""
Dataset adapters and DataLoader utilities for applying deterministic synthetic image corruptions.
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

from edge_vision.corruptions.registry import apply_corruption
from edge_vision.data.dataset import TransformedSubset


def _resolve_canonical_sample(dataset: Any, idx: int) -> Tuple[np.ndarray, int, int]:
    """
    Recursively traverses dataset subset wrappers (TransformedSubset, torch.utils.data.Subset)
    down to the underlying raw CIFAR-10 storage, returning:
      (raw_uint8_image_array, integer_target_label, canonical_dataset_index).

    Supported base dataset structures:
      1. torchvision.datasets.CIFAR10 or synthetic equivalents with .data (uint8 ndarray) and .targets.
      2. TransformedSubset wrapping a supported dataset.
      3. torch.utils.data.Subset wrapping a supported dataset.
      4. Nested combinations of the above.

    Explicitly rejected:
      - Any dataset missing raw uint8 array storage (e.g. TensorDataset, pre-transformed datasets).
    """
    if hasattr(dataset, "indices") and hasattr(dataset, "base_dataset"):
        # TransformedSubset
        real_idx = int(dataset.indices[idx])
        return _resolve_canonical_sample(dataset.base_dataset, real_idx)
    elif hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        # torch.utils.data.Subset
        real_idx = int(dataset.indices[idx])
        return _resolve_canonical_sample(dataset.dataset, real_idx)
    elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
        # Root raw dataset
        raw_data = dataset.data
        if not isinstance(raw_data, np.ndarray):
            raise TypeError(
                f"CorruptedDataset expects base dataset .data to be a numpy.ndarray, got {type(raw_data).__name__}"
            )
        if raw_data.dtype != np.uint8:
            raise TypeError(
                f"CorruptedDataset expects base dataset .data to have dtype uint8, got {raw_data.dtype}"
            )
        raw_img = raw_data[idx]
        target = int(dataset.targets[idx])
        return raw_img, target, int(idx)
    else:
        raise TypeError(
            f"CorruptedDataset requires a CIFAR-10-compatible dataset exposing raw uint8 .data "
            f"and .targets, or an indexed Subset over such a dataset. Received unsupported type: "
            f"{type(dataset).__name__}."
        )


class CorruptedDataset(Dataset):
    """
    Lightweight Dataset adapter that wraps a CIFAR-10 dataset or Subset view,
    applying deterministic synthetic corruption to the raw uint8 RGB image array
    BEFORE applying evaluation transforms (such as ToTensor and Normalize).

    Guarantees:
      - Memory efficiency: Shares underlying raw dataset array without duplication.
      - Non-destructive: Never modifies base_dataset.data in-place.
      - Pre-normalization: Corruption occurs on uint8 array [0, 255] before ToTensor / Normalize.
      - Stable sample identity: Resolves canonical dataset index for deterministic noise seeding.
      - Idempotent: Repeated __getitem__ calls for the same index yield bitwise identical outputs.
      - Explicit scope: Validates that underlying dataset provides raw uint8 arrays, rejecting
        unsupported or pre-transformed datasets with a clear TypeError.
    """

    def __init__(
        self,
        base_dataset: Union[Dataset, TransformedSubset, Subset],
        corruption_name: str,
        severity: int,
        transform: Optional[Any] = None,
        seed: int = 42,
    ):
        """
        Args:
            base_dataset: Raw CIFAR-10 dataset or indexed Subset/TransformedSubset view.
            corruption_name: Registered corruption name (e.g. 'gaussian_noise').
            severity: Severity level (integer in [1, 5]).
            transform: Evaluation transform to apply after corruption (e.g. ToTensor + Normalize).
            seed: Experiment seed for deterministic noise generation.
        """
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raw_img, target, canonical_idx = _resolve_canonical_sample(self.base_dataset, idx)

        # 1. Apply corruption to raw RGB array (H, W, 3) before normalization
        corrupted_arr = apply_corruption(
            image=raw_img,
            name=self.corruption_name,
            severity=self.severity,
            seed=self.seed,
            sample_idx=canonical_idx,
        )

        # 2. Convert corrupted array to PIL Image for torchvision transform compatibility
        pil_img = Image.fromarray(corrupted_arr)

        # 3. Apply evaluation transform (typically ToTensor + Normalize)
        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.functional.to_tensor(pil_img)

        return tensor_img, target


def get_corrupted_dataloader(
    base_dataset: Union[Dataset, TransformedSubset, Subset],
    corruption_name: str,
    severity: int,
    transform: Optional[Any] = None,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """
    Constructs a DataLoader over a CorruptedDataset.

    Args:
        base_dataset: Raw dataset or Subset view to corrupt.
        corruption_name: Registered corruption name.
        severity: Severity level (1-5).
        transform: Downstream evaluation transform.
        seed: Random seed for noise generation.
        batch_size: Batch size.
        num_workers: Worker count.

    Returns:
        DataLoader ready for corrupted evaluation.
    """
    dataset = CorruptedDataset(
        base_dataset=base_dataset,
        corruption_name=corruption_name,
        severity=severity,
        transform=transform,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
