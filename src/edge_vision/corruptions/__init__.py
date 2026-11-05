"""
Deterministic synthetic image corruption engine for robustness evaluation.
"""

from edge_vision.corruptions.config import (
    CorruptionSuiteConfig,
    SingleCorruptionConfig,
    get_default_corruption_config_path,
    load_corruption_config,
)
from edge_vision.corruptions.dataset import CorruptedDataset, get_corrupted_dataloader
from edge_vision.corruptions.deterministic import derive_deterministic_seed
from edge_vision.corruptions.operations import (
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_motion_blur,
    validate_image_array,
)
from edge_vision.corruptions.registry import (
    CORRUPTION_REGISTRY,
    CorruptionDefinition,
    apply_corruption,
    build_corruption_registry,
    get_corruption,
    get_severity_parameters,
    list_corruptions,
    reload_registry,
)

__all__ = [
    "CORRUPTION_REGISTRY",
    "CorruptionDefinition",
    "CorruptionSuiteConfig",
    "SingleCorruptionConfig",
    "CorruptedDataset",
    "apply_brightness",
    "apply_contrast",
    "apply_corruption",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_jpeg_compression",
    "apply_motion_blur",
    "build_corruption_registry",
    "derive_deterministic_seed",
    "get_corruption",
    "get_corrupted_dataloader",
    "get_default_corruption_config_path",
    "get_severity_parameters",
    "list_corruptions",
    "load_corruption_config",
    "reload_registry",
    "validate_image_array",
]
