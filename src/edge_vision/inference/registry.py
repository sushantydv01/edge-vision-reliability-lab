"""
Centralized artifact registry for resolving and validating inference model checkpoints.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from edge_vision.inference.exceptions import (
    InferenceError,
    ModelNotFoundError,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)

# Canonical architectures and human-readable aliases
CANONICAL_MODELS = {
    "resnet18_cifar": "resnet18_cifar",
    "resnet18": "resnet18_cifar",
    "mobilenet_v3_small_cifar": "mobilenet_v3_small_cifar",
    "mobilenet_v3_small": "mobilenet_v3_small_cifar",
    "mobilenet": "mobilenet_v3_small_cifar",
}

CANONICAL_PRECISIONS = {"fp32", "fp16", "int8"}

# Authoritative parameter counts established in Task 2.1 & 5.2B
MODEL_PARAMETERS = {
    "resnet18_cifar": 11_173_962,
    "mobilenet_v3_small_cifar": 1_528_106,
}

# Canonical relative paths from project root
DEFAULT_ARTIFACT_REL_PATHS: Dict[Tuple[str, str], str] = {
    ("resnet18_cifar", "fp32"): "experiments/checkpoints_inference_fp32/resnet18_cifar_fp32.pt",
    ("resnet18_cifar", "fp16"): "experiments/checkpoints_fp16/resnet18_cifar_fp16.pt",
    ("resnet18_cifar", "int8"): "experiments/checkpoints_int8/resnet18_cifar_int8.pt",
    ("mobilenet_v3_small_cifar", "fp32"): "experiments/checkpoints_inference_fp32/mobilenet_v3_small_cifar_fp32.pt",
    ("mobilenet_v3_small_cifar", "fp16"): "experiments/checkpoints_fp16/mobilenet_v3_small_cifar_fp16.pt",
    ("mobilenet_v3_small_cifar", "int8"): "experiments/checkpoints_int8/mobilenet_v3_small_cifar_int8.pt",
}


def find_project_root() -> Path:
    """
    Locates the project root directory deterministically by inspecting parents
    for marker files (pyproject.toml).
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ArtifactInfo:
    """
    Verified metadata describing a resolved model artifact.
    """
    model_name: str
    precision: str
    artifact_path: Path
    relative_path: str
    file_size_bytes: int
    parameter_count: int
    companion_metadata: Optional[Dict[str, Any]] = None


class ArtifactRegistry:
    """
    Resolves canonical artifact paths and loads accompanying metadata.
    """

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        """Normalizes and validates model architecture name."""
        key = model_name.strip().lower()
        if key not in CANONICAL_MODELS:
            supported = sorted(list(set(CANONICAL_MODELS.values())))
            raise UnsupportedModelError(
                f"Unsupported model architecture '{model_name}'. Supported architectures: {supported}."
            )
        return CANONICAL_MODELS[key]

    @classmethod
    def normalize_precision(cls, precision: str) -> str:
        """Normalizes and validates precision string."""
        prec = precision.strip().lower()
        if prec not in CANONICAL_PRECISIONS:
            supported = sorted(list(CANONICAL_PRECISIONS))
            raise UnsupportedPrecisionError(
                f"Unsupported precision '{precision}'. Supported precisions: {supported}."
            )
        return prec

    @classmethod
    def list_supported_configurations(cls) -> List[Tuple[str, str]]:
        """Returns sorted list of all supported (model_name, precision) pairs."""
        return sorted(list(DEFAULT_ARTIFACT_REL_PATHS.keys()))

    @classmethod
    def resolve(
        cls,
        model_name: str,
        precision: str = "fp32",
        checkpoint_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> ArtifactInfo:
        """
        Resolves the artifact path for a given architecture and precision,
        validating that the file exists and verifying companion metadata.

        Parameters:
            model_name: Architecture name or alias (e.g. 'resnet18_cifar').
            precision: Model precision ('fp32', 'fp16', 'int8').
            checkpoint_path: Optional path to override default canonical location.
            project_root: Optional root directory override.

        Returns:
            ArtifactInfo containing resolved absolute path, size, and metadata.

        Raises:
            UnsupportedModelError: If model_name is invalid.
            UnsupportedPrecisionError: If precision is invalid.
            ModelNotFoundError: If artifact file is missing on disk.
        """
        norm_model = cls.normalize_model_name(model_name)
        norm_prec = cls.normalize_precision(precision)
        root = Path(project_root).resolve() if project_root else find_project_root()

        if checkpoint_path is not None:
            candidate = Path(checkpoint_path)
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()
            rel_str = str(candidate)
        else:
            rel_str = DEFAULT_ARTIFACT_REL_PATHS[(norm_model, norm_prec)]
            candidate = (root / rel_str).resolve()

        if not candidate.is_file():
            raise ModelNotFoundError(
                f"Model artifact not found for {norm_model} [{norm_prec}] at '{candidate}'."
            )

        file_sz = candidate.stat().st_size
        param_count = MODEL_PARAMETERS.get(norm_model, 0)

        # Load companion JSON if available (mandatory verification for INT8)
        companion_meta: Optional[Dict[str, Any]] = None
        companion_path = candidate.with_suffix(".json")
        if companion_path.is_file():
            try:
                with open(companion_path, "r", encoding="utf-8") as f:
                    companion_meta = json.load(f)
            except Exception as err:
                raise InferenceError(
                    f"Failed to read companion metadata JSON at '{companion_path}': {err}"
                ) from err

            # Validate integrity of companion JSON against requested model and precision
            if not isinstance(companion_meta, dict):
                raise InferenceError(
                    f"Malformed companion metadata JSON at '{companion_path}': expected JSON object dictionary."
                )
            meta_prec = str(companion_meta.get("precision", "")).strip().lower()
            meta_arch = str(companion_meta.get("model_architecture", "")).strip()
            if meta_prec and meta_prec != norm_prec:
                raise InferenceError(
                    f"Companion metadata precision mismatch: expected '{norm_prec}', got '{meta_prec}' in '{companion_path}'."
                )
            if meta_arch and meta_arch != norm_model:
                raise InferenceError(
                    f"Companion metadata architecture mismatch: expected '{norm_model}', got '{meta_arch}' in '{companion_path}'."
                )

        return ArtifactInfo(
            model_name=norm_model,
            precision=norm_prec,
            artifact_path=candidate,
            relative_path=rel_str,
            file_size_bytes=file_sz,
            parameter_count=param_count,
            companion_metadata=companion_meta,
        )
