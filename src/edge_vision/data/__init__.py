"""
CIFAR-10 data pipeline modules for Edge Vision Reliability Lab.
"""

from edge_vision.data.config import (
    DataPipelineConfig,
    DatasetConfig,
    SplitConfig,
    NormalizationConfig,
    AugmentationConfig,
    DataLoaderConfig,
    load_data_config,
)
from edge_vision.data.transforms import get_cifar10_transforms
from edge_vision.data.splits import (
    compute_indices_checksum,
    validate_split_indices,
    validate_manifest_schema,
    generate_train_val_split_indices,
    save_split_manifest,
    load_split_manifest,
    get_or_create_split_indices,
)
from edge_vision.data.dataset import (
    CIFAR10_CLASSES,
    TransformedSubset,
    CIFAR10DataLoaders,
    get_cifar10_dataloaders,
)

__all__ = [
    "DataPipelineConfig",
    "DatasetConfig",
    "SplitConfig",
    "NormalizationConfig",
    "AugmentationConfig",
    "DataLoaderConfig",
    "load_data_config",
    "get_cifar10_transforms",
    "compute_indices_checksum",
    "validate_split_indices",
    "validate_manifest_schema",
    "generate_train_val_split_indices",
    "save_split_manifest",
    "load_split_manifest",
    "get_or_create_split_indices",
    "CIFAR10_CLASSES",
    "TransformedSubset",
    "CIFAR10DataLoaders",
    "get_cifar10_dataloaders",
]
