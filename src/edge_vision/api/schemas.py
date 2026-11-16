"""
Pydantic response and request schemas for the REST inference API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(default="healthy", description="Service health status.")
    model_loaded: bool = Field(description="Whether an active inference model is currently loaded in memory.")
    model_name: Optional[str] = Field(default=None, description="Architecture name of loaded model, if any.")
    precision: Optional[str] = Field(default=None, description="Precision variant of loaded model, if any.")
    device: Optional[str] = Field(default=None, description="Compute device of loaded model, if any.")
    backend: Optional[str] = Field(default=None, description="Execution backend (e.g. pytorch_eager, torchscript).")


class ModelArtifactSummary(BaseModel):
    """Summary of a supported model artifact configuration from the registry."""
    model_name: str = Field(description="Canonical model architecture identifier.")
    precision: str = Field(description="Precision variant (fp32, fp16, int8).")
    supported_backends: List[str] = Field(description="Supported execution backends.")
    preferred_device: str = Field(description="Default or preferred execution device.")
    file_size_bytes: int = Field(description="Artifact file size on disk in bytes.")
    parameter_count: int = Field(description="Total model parameter count.")


class ModelsListResponse(BaseModel):
    """Response containing all supported model configurations in the registry."""
    models: List[ModelArtifactSummary] = Field(description="List of available model configurations.")
    total_count: int = Field(description="Total number of supported configurations.")


class CurrentModelResponse(BaseModel):
    """Detailed metadata for the currently active inference model."""
    loaded: bool = Field(description="Whether a model is currently loaded.")
    model_name: Optional[str] = Field(default=None, description="Active model architecture.")
    precision: Optional[str] = Field(default=None, description="Active precision variant.")
    device: Optional[str] = Field(default=None, description="Active execution device.")
    backend: Optional[str] = Field(default=None, description="Active runtime backend.")
    parameter_count: Optional[int] = Field(default=None, description="Parameter count of active model.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional model and companion metadata.")


class ModelLoadRequest(BaseModel):
    """Request payload to load or switch the active inference model."""
    model_name: str = Field(description="Model architecture identifier (e.g. resnet18_cifar, mobilenet_v3_small_cifar).")
    precision: str = Field(description="Target precision variant (fp32, fp16, int8).")
    device: Optional[str] = Field(default=None, description="Optional target device (e.g. cpu, mps).")


class ModelLoadResponse(BaseModel):
    """Response payload confirming successful model initialization."""
    status: str = Field(default="success", description="Status indicator.")
    message: str = Field(description="Human-readable confirmation message.")
    model_name: str = Field(description="Loaded model architecture.")
    precision: str = Field(description="Loaded precision variant.")
    device: str = Field(description="Resolved execution device.")
    backend: str = Field(description="Resolved runtime backend.")
    load_time_ms: float = Field(description="Time taken to load and initialize the model in milliseconds.")


class TopKPrediction(BaseModel):
    """A single rank-k class prediction entry."""
    rank: int = Field(description="1-based rank of the prediction.")
    class_name: str = Field(description="CIFAR-10 class label.")
    confidence: float = Field(description="Softmax probability in [0.0, 1.0].")


class PredictionDetail(BaseModel):
    """Comprehensive classification prediction output."""
    predicted_class: str = Field(description="Highest-confidence predicted class label.")
    class_index: int = Field(description="0-based CIFAR-10 class index [0, 9].")
    confidence: float = Field(description="Softmax confidence score for top prediction.")
    probabilities: Dict[str, float] = Field(description="Full probability distribution over 10 CIFAR-10 classes.")
    logits: List[float] = Field(description="Raw unnormalized model output logits across 10 CIFAR-10 classes.")
    top_k: List[TopKPrediction] = Field(description="Top-k highest-confidence predictions in descending order.")


class InferenceTimingSchema(BaseModel):
    """Component latency breakdown for inference execution."""
    preprocess_ms: float = Field(description="Time spent validating, resizing, and normalizing inputs in ms.")
    forward_ms: float = Field(description="Time spent on tensor transfer, forward execution, and device sync in ms.")
    postprocess_ms: float = Field(description="Time spent on softmax calculation and prediction formatting in ms.")
    total_ms: float = Field(description="Total execution latency in ms.")


class SinglePredictionResponse(BaseModel):
    """Response payload for single image inference."""
    status: str = Field(default="success", description="Status indicator.")
    filename: str = Field(description="Original uploaded filename.")
    prediction: PredictionDetail = Field(description="Classification prediction results.")
    timing_ms: InferenceTimingSchema = Field(description="Latency breakdown in milliseconds.")
    model_name: str = Field(description="Architecture used for prediction.")
    precision: str = Field(description="Precision used for prediction.")
    device: str = Field(description="Device used for prediction.")


class BatchItemPrediction(BaseModel):
    """Prediction container for one image within a batch request."""
    filename: str = Field(description="Original uploaded filename.")
    index: int = Field(description="0-based index preserving input order.")
    prediction: PredictionDetail = Field(description="Classification prediction results.")


class BatchPredictionResponse(BaseModel):
    """Response payload for bounded batch inference."""
    status: str = Field(default="success", description="Status indicator.")
    batch_size: int = Field(description="Total number of images evaluated in the batch.")
    predictions: List[BatchItemPrediction] = Field(description="Ordered list of per-image predictions.")
    timing_ms: InferenceTimingSchema = Field(description="Aggregated batch execution latency breakdown in ms.")
    throughput_fps: float = Field(description="Observed throughput in images per second.")
    model_name: str = Field(description="Architecture used for prediction.")
    precision: str = Field(description="Precision used for prediction.")
    device: str = Field(description="Device used for prediction.")


class ErrorResponse(BaseModel):
    """Structured error response format."""
    detail: str = Field(description="Human-readable description of the error.")
    error_type: str = Field(description="Categorical name of the raised exception or error condition.")
