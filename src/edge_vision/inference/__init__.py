"""
Inference and deployment layer for Edge Vision Reliability Lab.
Provides unified inference engine, preprocessor, artifact registry, and structured output types.
"""

from edge_vision.inference.exceptions import (
    DeviceCompatibilityError,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)
from edge_vision.inference.types import (
    BatchInferenceOutput,
    InferenceTiming,
    PredictionResult,
    SingleImageInferenceOutput,
)
from edge_vision.inference.preprocessing import ImagePreprocessor
from edge_vision.inference.registry import (
    ArtifactInfo,
    ArtifactRegistry,
)
from edge_vision.inference.engine import InferenceEngine

__all__ = [
    # Types
    "PredictionResult",
    "InferenceTiming",
    "SingleImageInferenceOutput",
    "BatchInferenceOutput",
    # Preprocessing
    "ImagePreprocessor",
    # Registry
    "ArtifactInfo",
    "ArtifactRegistry",
    # Engine
    "InferenceEngine",
    # Exceptions
    "InferenceError",
    "ModelNotFoundError",
    "UnsupportedModelError",
    "UnsupportedPrecisionError",
    "DeviceCompatibilityError",
    "InvalidInputError",
]
