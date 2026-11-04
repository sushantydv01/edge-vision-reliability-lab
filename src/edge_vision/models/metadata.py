"""
Model parameter counting and memory footprint utilities.
"""

from typing import Dict, Union
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Counts total, trainable, and non-trainable parameters in a PyTorch module.

    Parameters:
        model: PyTorch model instance.

    Returns:
        Dictionary with keys:
          - 'total': Total parameter count.
          - 'trainable': Count of parameters requiring gradients.
          - 'non_trainable': Count of frozen/non-trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def compute_parameter_memory(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calculates parameter memory footprint directly from parameter tensors and dtypes.

    Parameters:
        model: PyTorch model instance.

    Returns:
        Dictionary with keys:
          - 'bytes': Total bytes allocated for parameters.
          - 'megabytes_decimal': Memory size in decimal megabytes (MB, 10^6 bytes).
          - 'mebibytes_binary': Memory size in binary mebibytes (MiB, 2^20 bytes).
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "bytes": total_bytes,
        "megabytes_decimal": total_bytes / 1_000_000,
        "mebibytes_binary": total_bytes / (1024 * 1024),
    }


def get_model_summary(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Aggregates parameter counts and memory footprint metrics for a model.

    Parameters:
        model: PyTorch model instance.

    Returns:
        Combined dictionary containing parameter counts and memory footprint metrics.
    """
    params = count_parameters(model)
    memory = compute_parameter_memory(model)
    return {
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "non_trainable_parameters": params["non_trainable"],
        "parameter_bytes": memory["bytes"],
        "parameter_mb": memory["megabytes_decimal"],
        "parameter_mib": memory["mebibytes_binary"],
    }
