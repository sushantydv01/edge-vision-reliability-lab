"""
Model registry and factory for CIFAR-10 architectures.
"""

from typing import Any, Callable, Dict, List
import torch.nn as nn

from edge_vision.models.resnet import resnet18_cifar
from edge_vision.models.mobilenet import mobilenet_v3_small_cifar

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18_cifar": resnet18_cifar,
    "mobilenet_v3_small_cifar": mobilenet_v3_small_cifar,
}


def list_available_models() -> List[str]:
    """
    Returns a sorted list of registered model architecture names.
    """
    return sorted(list(MODEL_REGISTRY.keys()))


def get_model(name: str, num_classes: int = 10, **kwargs: Any) -> nn.Module:
    """
    Instantiates an architecture by its registered name.

    Parameters:
        name: Architecture name (e.g., 'resnet18_cifar', 'mobilenet_v3_small_cifar').
        num_classes: Number of output classification logits (default: 10).
        **kwargs: Additional keyword arguments forwarded to the model constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = list_available_models()
        raise KeyError(
            f"Unknown model architecture '{name}'. Available architectures: {available}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
