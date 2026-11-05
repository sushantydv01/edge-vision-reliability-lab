"""
Corruption registry, metadata lookup, and unified dispatch interface.

YAML is the single authoritative source for corruption severity parameters.
No duplicate parameter tables exist in Python code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from edge_vision.corruptions.config import load_corruption_config
from edge_vision.corruptions.operations import (
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_motion_blur,
    validate_image_array,
)


@dataclass(frozen=True)
class CorruptionDefinition:
    """Specification and execution metadata for a single corruption type."""
    name: str
    description: str
    parameter_name: str
    is_stochastic: bool
    severities: Dict[int, Union[int, float]]
    fn: Callable[..., np.ndarray]


# Static binding of corruption names to pure mathematical operations and stochastic nature
_CORRUPTION_OPS: Dict[str, Tuple[Callable[..., np.ndarray], bool]] = {
    "gaussian_noise": (apply_gaussian_noise, True),
    "gaussian_blur": (apply_gaussian_blur, False),
    "motion_blur": (apply_motion_blur, False),
    "brightness": (apply_brightness, False),
    "contrast": (apply_contrast, False),
    "jpeg_compression": (apply_jpeg_compression, False),
}


def build_corruption_registry(config_path: Optional[Union[str, Path]] = None) -> Dict[str, CorruptionDefinition]:
    """
    Constructs the runtime corruption registry from authoritative YAML configuration.
    Guarantees that no duplicate parameter tables exist in Python source code.

    Args:
        config_path: Optional path to YAML configuration. If None, resolves the default.

    Returns:
        Dictionary mapping corruption names to fully populated CorruptionDefinition instances.
    """
    suite_cfg = load_corruption_config(config_path)
    registry: Dict[str, CorruptionDefinition] = {}

    for name, (fn, is_stochastic) in _CORRUPTION_OPS.items():
        if name not in suite_cfg.corruptions:
            raise ValueError(f"Required corruption operation '{name}' not found in configuration")

        c_cfg = suite_cfg.corruptions[name]
        registry[name] = CorruptionDefinition(
            name=name,
            description=c_cfg.description,
            parameter_name=c_cfg.parameter_name,
            is_stochastic=is_stochastic,
            severities=dict(c_cfg.severities),
            fn=fn,
        )

    return registry


# Initial runtime registry populated dynamically from authoritative YAML configuration
CORRUPTION_REGISTRY: Dict[str, CorruptionDefinition] = build_corruption_registry()


def reload_registry(config_path: Optional[Union[str, Path]] = None) -> None:
    """
    Reloads the global CORRUPTION_REGISTRY from a specified or default YAML config.
    """
    global CORRUPTION_REGISTRY
    CORRUPTION_REGISTRY = build_corruption_registry(config_path)


def list_corruptions() -> List[str]:
    """Returns a sorted list of registered corruption names."""
    return sorted(CORRUPTION_REGISTRY.keys())


def get_corruption(name: str) -> CorruptionDefinition:
    """
    Retrieves the definition of a registered corruption by name.

    Args:
        name: Corruption name identifier.

    Returns:
        CorruptionDefinition instance.

    Raises:
        ValueError: If name is not registered.
    """
    if name not in CORRUPTION_REGISTRY:
        available = ", ".join(repr(k) for k in list_corruptions())
        raise ValueError(f"Unknown corruption '{name}'. Available corruptions: {available}")
    return CORRUPTION_REGISTRY[name]


def get_severity_parameters(
    name: str,
    severity: Optional[int] = None,
) -> Union[Dict[int, Union[int, float]], Union[int, float]]:
    """
    Retrieves severity parameter(s) for a given corruption.

    Args:
        name: Corruption name.
        severity: Optional severity level (1-5). If None, returns dict of all severities.

    Returns:
        Parameter value for the severity level, or dict mapping severities to values.
    """
    defn = get_corruption(name)
    if severity is None:
        return dict(defn.severities)

    if isinstance(severity, bool) or not isinstance(severity, int) or severity < 1 or severity > 5:
        raise ValueError(f"Severity must be an integer between 1 and 5, got {severity!r}")

    if severity not in defn.severities:
        raise ValueError(f"Severity {severity} not configured for corruption '{name}'")

    return defn.severities[severity]


def apply_corruption(
    image: np.ndarray,
    name: str,
    severity: int,
    seed: Optional[int] = None,
    sample_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Applies a named synthetic corruption operation at a specified severity level.

    Args:
        image: Raw RGB NumPy array of shape (H, W, 3) and dtype uint8.
        name: Registered corruption name.
        severity: Severity level (integer in [1, 5]).
        seed: Random seed for stochastic operations (e.g. gaussian_noise).
        sample_idx: Canonical index of the image in the dataset for deterministic noise seeding.

    Returns:
        New corrupted RGB NumPy array of shape (H, W, 3) and dtype uint8.

    Raises:
        TypeError: If image is not a NumPy array or has incorrect dtype.
        ValueError: If image dimensions are invalid, corruption name is unknown,
                    or severity is outside [1, 5].
    """
    validate_image_array(image)
    defn = get_corruption(name)

    if isinstance(severity, bool) or not isinstance(severity, int) or severity < 1 or severity > 5:
        raise ValueError(f"Severity must be an integer between 1 and 5, got {severity!r}")

    param = defn.severities[severity]

    if defn.is_stochastic:
        return defn.fn(
            image=image,
            std_norm=param,
            rng=None,
            seed=seed,
            sample_idx=sample_idx,
            severity=severity,
        )
    else:
        return defn.fn(image, param)
