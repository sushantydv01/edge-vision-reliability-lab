"""
Structured data types and containers for inference outputs and profiling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PredictionResult:
    """
    Structured classification prediction for a single image.

    Attributes:
        predicted_class: Human-readable class name string (e.g. 'airplane').
        class_index: Target class index integer in [0, num_classes - 1].
        confidence: Normalized probability in [0.0, 1.0] of predicted class.
        probabilities: Dictionary mapping every class name to its softmax probability.
        logits: Raw unnormalized model output logits across classes.
        top_k: Optional sorted list of (class_name, probability) tuples for top-k predictions.
    """
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    logits: List[float]
    top_k: Optional[List[Tuple[str, float]]] = None


@dataclass(frozen=True)
class InferenceTiming:
    """
    Latency profiling decomposition measured in milliseconds (ms).

    Attributes:
        preprocess_ms: Time spent converting, validating, resizing, and normalizing input.
        forward_ms: Time spent executing model forward pass (including device synchronization).
        postprocess_ms: Time spent applying softmax, argmax, and formatting outputs.
        total_ms: Total end-to-end request latency (preprocess + forward + postprocess).
    """
    preprocess_ms: float
    forward_ms: float
    postprocess_ms: float
    total_ms: float


@dataclass(frozen=True)
class SingleImageInferenceOutput:
    """
    Complete inference response for a single image input.

    Attributes:
        prediction: PredictionResult instance.
        timing: InferenceTiming decomposition.
        metadata: Context dictionary (model_architecture, precision, device, input_shape, etc.).
    """
    prediction: PredictionResult
    timing: InferenceTiming
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class BatchInferenceOutput:
    """
    Complete inference response for a batch of image inputs.

    Attributes:
        predictions: List of PredictionResult instances, one per input sample.
        timing: Aggregated InferenceTiming decomposition for the entire batch.
        batch_size: Number of images in the batch.
        throughput_fps: Calculated throughput in images per second (batch_size / (total_ms / 1000.0)).
        metadata: Context dictionary (model_architecture, precision, device, batch_shape, etc.).
    """
    predictions: List[PredictionResult]
    timing: InferenceTiming
    batch_size: int
    throughput_fps: float
    metadata: Dict[str, Any]
