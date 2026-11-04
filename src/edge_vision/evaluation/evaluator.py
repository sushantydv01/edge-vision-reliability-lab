"""
Standalone evaluation execution, checkpoint model reconstruction, and atomic report persistence.
"""

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


def load_checkpoint_for_evaluation(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads and reconstructs an evaluation-ready model from a previously saved checkpoint.

    Guarantees:
      1. Reconstructs architecture via existing model factory registry.
      2. Strictly loads the model state dictionary (strict=True).
      3. Sets model.eval().
      4. Never mutates the checkpoint on disk.

    Parameters:
        checkpoint_path: Path to checkpoint .pt file.
        device: Optional torch.device to move model to.

    Returns:
        Tuple of (model in eval mode, raw checkpoint payload dictionary).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = CheckpointManager.load_checkpoint(path)

    if "model_architecture" not in checkpoint:
        raise ValueError(f"Malformed checkpoint at {path}: missing 'model_architecture' metadata.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Malformed checkpoint at {path}: missing 'model_state_dict'.")

    arch_name = checkpoint["model_architecture"]
    model = get_model(arch_name, num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
    model.to(device)
    tracker = EvaluationTracker(class_names=class_names)

    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

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
