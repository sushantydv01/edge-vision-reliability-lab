"""
Model architectures and model factory for Edge Vision Reliability Lab.
"""

from edge_vision.models.resnet import resnet18_cifar
from edge_vision.models.mobilenet import mobilenet_v3_small_cifar
from edge_vision.models.metadata import (
    count_parameters,
    compute_parameter_memory,
    get_model_summary,
)
from edge_vision.models.factory import (
    MODEL_REGISTRY,
    get_model,
    list_available_models,
)

__all__ = [
    "resnet18_cifar",
    "mobilenet_v3_small_cifar",
    "count_parameters",
    "compute_parameter_memory",
    "get_model_summary",
    "MODEL_REGISTRY",
    "get_model",
    "list_available_models",
]
