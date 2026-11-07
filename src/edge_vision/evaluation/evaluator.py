"""
Standalone evaluation execution, checkpoint model reconstruction, and atomic report persistence.
"""

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets

from edge_vision.data.config import load_data_config
from edge_vision.data.dataset import CIFAR10_CLASSES, TransformedSubset
from edge_vision.data.splits import get_or_create_split_indices
from edge_vision.data.transforms import get_cifar10_transforms
from edge_vision.evaluation.metrics import EvaluationMetrics, EvaluationTracker
from edge_vision.models.factory import get_model
from edge_vision.training.checkpoint import CheckpointManager


def resolve_evaluation_device(device_req: str) -> torch.device:
    """
    Resolves the execution device for evaluation.
    Rejects unavailable explicitly requested devices with RuntimeError.
    """
    req = device_req.strip().lower()
    if req == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif req == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested for evaluation but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    elif req == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported evaluation device '{device_req}'. Expected 'auto', 'cpu', or 'mps'.")


def convert_model_to_fp16(model: nn.Module) -> nn.Module:
    """
    Converts a model to half precision (FP16) in-place and sets eval mode.
    Freezes all parameters and validates that model parameters and floating-point
    buffers are successfully cast to torch.float16.

    Parameters:
        model: PyTorch nn.Module instance.

    Returns:
        The half-precision model in eval mode.
    """
    model.half()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        if param.dtype != torch.float16:
            raise TypeError(f"Failed to convert parameter to float16, got {param.dtype}")
    for name, buf in model.named_buffers():
        if buf.is_floating_point() and buf.dtype != torch.float16:
            raise TypeError(f"Failed to convert buffer {name} to float16, got {buf.dtype}")
    return model


def export_inference_checkpoint(
    model: nn.Module,
    model_architecture: str,
    output_path: Union[str, Path],
    precision: str = "fp32",
    num_classes: int = 10,
    source_checkpoint_path: Optional[Union[str, Path]] = None,
    source_checkpoint_sha256: Optional[str] = None,
    source_epoch: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically exports a lightweight, inference-only model artifact.
    Contains strictly model parameters and deployment metadata (no optimizer or RNG states).

    Parameters:
        model: Evaluated PyTorch model.
        model_architecture: Architecture identifier string.
        output_path: Destination path for the .pt artifact.
        precision: 'fp32' or 'fp16'.
        num_classes: Number of classification target classes (default 10).
        source_checkpoint_path: Optional path to the source training checkpoint.
        source_checkpoint_sha256: Optional SHA-256 hash of the source checkpoint.
        source_epoch: Optional epoch number of the source checkpoint.
        extra_metadata: Optional additional key-value metadata to persist.

    Returns:
        Path to the saved artifact.
    """
    prec_norm = precision.strip().lower()
    if prec_norm not in ("fp32", "fp16"):
        raise ValueError(f"Unsupported precision '{precision}'. Expected 'fp32' or 'fp16'.")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(f".tmp.{os.getpid()}")

    # Prepare model state dict in target precision
    if prec_norm == "fp16":
        export_model = copy.deepcopy(model)
        convert_model_to_fp16(export_model)
        export_state_dict = export_model.state_dict()
    else:
        export_model = copy.deepcopy(model).float()
        export_state_dict = export_model.state_dict()

    param_count = sum(p.numel() for p in export_model.parameters())

    payload: Dict[str, Any] = {
        "model_architecture": model_architecture,
        "precision": prec_norm,
        "num_classes": num_classes,
        "parameter_count": param_count,
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_epoch": source_epoch,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": export_state_dict,
    }
    if extra_metadata:
        payload["extra_metadata"] = extra_metadata

    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, dest)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return dest


def load_checkpoint_for_evaluation(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    precision: str = "auto",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads and reconstructs an evaluation-ready model from a previously saved checkpoint.

    Guarantees:
      1. Reconstructs architecture via existing model factory registry.
      2. Strictly loads the model state dictionary (strict=True).
      3. Sets model.eval().
      4. Converts to half precision (FP16) if requested or if checkpoint is native FP16.
      5. Never mutates the checkpoint on disk.

    Parameters:
        checkpoint_path: Path to checkpoint .pt file.
        device: Optional torch.device to move model to.
        precision: Target precision ('auto', 'fp32', or 'fp16'). Default 'auto' preserves
                   checkpoint native precision (or fp32 if unspecified in checkpoint).

    Returns:
        Tuple of (model in eval mode, raw checkpoint payload dictionary).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    prec_norm = precision.strip().lower()
    if prec_norm not in ("auto", "fp32", "fp16", "int8"):
        raise ValueError(f"Unsupported precision '{precision}'. Expected 'auto', 'fp32', 'fp16', or 'int8'.")

    # TorchScript artifact loading (e.g. INT8 models)
    if prec_norm == "int8" or str(path).endswith("_int8.pt"):
        torch.backends.quantized.engine = "qnnpack"
        eval_device = device if device is not None else torch.device("cpu")
        model = torch.jit.load(str(path), map_location=eval_device)
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {
                "precision": "int8",
                "model_architecture": "resnet18_cifar" if "resnet" in path.name else "mobilenet_v3_small_cifar",
            }
        return model, checkpoint

    checkpoint = CheckpointManager.load_checkpoint(path)

    if "model_architecture" not in checkpoint:
        raise ValueError(f"Malformed checkpoint at {path}: missing 'model_architecture' metadata.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Malformed checkpoint at {path}: missing 'model_state_dict'.")

    ckpt_precision = checkpoint.get("precision", "fp32")
    if prec_norm == "auto":
        target_precision = ckpt_precision
    else:
        target_precision = prec_norm

    arch_name = checkpoint["model_architecture"]
    num_classes = checkpoint.get("num_classes", 10)
    model = get_model(arch_name, num_classes=num_classes)

    # If checkpoint state_dict has fp16 weights, convert model first so load_state_dict matches dtypes
    if ckpt_precision == "fp16":
        convert_model_to_fp16(model)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if target_precision == "fp16":
        convert_model_to_fp16(model)
    elif target_precision == "fp32":
        model.float()

    model.eval()

    if device is not None:
        model.to(device)

    return model, checkpoint


def get_evaluation_dataloader(
    data_config_path: str = "configs/cifar10.yaml",
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """
    Constructs an evaluation DataLoader with strictly unaugmented transforms.

    Parameters:
        data_config_path: Path to dataset YAML configuration.
        split: 'test' for the held-out test set, or 'val' for the validation partition.
        batch_size: Batch size for evaluation.
        num_workers: Worker count (default 0).

    Returns:
        DataLoader configured for evaluation.
    """
    split_norm = split.strip().lower()
    if split_norm not in ("test", "val"):
        raise ValueError(f"Unsupported evaluation split '{split}'. Expected 'test' or 'val'.")

    cfg = load_data_config(data_config_path)

    # Strictly deterministic, unaugmented evaluation transform
    eval_transform = get_cifar10_transforms(
        train=False,
        augment=False,
        mean=cfg.normalization.mean,
        std=cfg.normalization.std,
    )

    if split_norm == "test":
        dataset = datasets.CIFAR10(
            root=cfg.dataset.raw_dir,
            train=False,
            download=cfg.dataset.download,
            transform=eval_transform,
        )
    else:  # val
        base_train_dataset = datasets.CIFAR10(
            root=cfg.dataset.raw_dir,
            train=True,
            download=cfg.dataset.download,
            transform=None,
        )
        _, val_idx = get_or_create_split_indices(
            manifest_path=cfg.split.manifest_path,
            total_samples=len(base_train_dataset),
            train_size=cfg.split.train_size,
            val_size=cfg.split.val_size,
            seed=cfg.split.seed,
        )
        dataset = TransformedSubset(base_train_dataset, val_idx, transform=eval_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str] = CIFAR10_CLASSES,
) -> EvaluationMetrics:
    """
    Executes a complete evaluation pass using model.eval() and torch.inference_mode().

    Parameters:
        model: Evaluated PyTorch model.
        dataloader: Evaluation DataLoader.
        criterion: Loss criterion (e.g. nn.CrossEntropyLoss()).
        device: Execution device.
        class_names: Sequence of class label names.

    Returns:
        EvaluationMetrics dataclass instance.
    """
    model.eval()
    if hasattr(model, "to"):
        model.to(device)
    tracker = EvaluationTracker(class_names=class_names)
    model_dtype = torch.float32
    for p in model.parameters():
        model_dtype = p.dtype
        break

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if inputs.dtype != model_dtype:
                inputs = inputs.to(dtype=model_dtype)

            outputs = model(inputs)
            loss = criterion(outputs.float(), targets)

            tracker.update(loss.item(), outputs, targets, batch_size=inputs.size(0))

    return tracker.compute()


def save_evaluation_report(
    report_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Atomically saves a structured evaluation report to a JSON file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(f".tmp.{os.getpid()}")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, sort_keys=True)
        os.replace(temp_path, dest)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return dest
