"""
CPU-based INT8 Post-Training Static Quantization (PTQ) pipeline using PyTorch FX and QNNPACK.
"""

import copy
from datetime import datetime, timezone
import json
import operator
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping, QConfigMapping
import torch.ao.quantization.quantize_fx as qfx
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from edge_vision.data.config import load_data_config
from edge_vision.data.dataset import get_or_create_split_indices
from edge_vision.data.transforms import get_cifar10_transforms


class _CalibrationSubset(Dataset):
    """Memory-conscious indexed subset applying clean evaluation transforms."""

    def __init__(self, base_dataset: Dataset, indices: List[int], transform: Any):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_calibration_dataloader(
    data_config_path: str = "configs/cifar10.yaml",
    num_samples: int = 512,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    """
    Constructs a deterministic calibration DataLoader using strictly the training partition.
    Guarantees that validation and test partitions are completely isolated and never accessed.

    Parameters:
        data_config_path: Path to dataset YAML configuration.
        num_samples: Exact number of calibration images to sample from the training set.
        batch_size: Batch size for calibration inference.
        seed: Random seed for deterministic sample selection.

    Returns:
        DataLoader providing clean, unaugmented calibration images.
    """
    cfg = load_data_config(data_config_path)

    # 1. Load raw CIFAR-10 training dataset
    raw_train = datasets.CIFAR10(
        root=cfg.dataset.raw_dir,
        train=True,
        download=cfg.dataset.download,
        transform=None,
    )

    # 2. Acquire deterministic split indices (strictly training partition)
    train_idx, val_idx = get_or_create_split_indices(
        manifest_path=cfg.split.manifest_path,
        total_samples=len(raw_train),
        train_size=cfg.split.train_size,
        val_size=cfg.split.val_size,
        seed=cfg.split.seed,
    )

    if num_samples > len(train_idx):
        raise ValueError(f"Requested {num_samples} calibration samples, but training split only has {len(train_idx)}.")

    # 3. Deterministically sample calibration indices from training split ONLY
    rng = np.random.RandomState(seed)
    chosen_indices = [int(train_idx[i]) for i in rng.choice(len(train_idx), size=num_samples, replace=False)]

    # 4. Strictly unaugmented, clean evaluation transform
    calib_transform = get_cifar10_transforms(
        train=False,
        augment=False,
        mean=cfg.normalization.mean,
        std=cfg.normalization.std,
    )

    dataset = _CalibrationSubset(raw_train, chosen_indices, calib_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def inspect_quantized_operators(converted_model: torch.fx.GraphModule) -> Dict[str, Any]:
    """
    Inspects an FX graph module to categorize all quantized and floating-point operations.

    Parameters:
        converted_model: PyTorch FX converted graph module.

    Returns:
        Dictionary inventory of modules and function calls in the graph.
    """
    modules: Dict[str, int] = {}
    functions: Dict[str, int] = {}

    for node in converted_model.graph.nodes:
        if node.op == "call_module":
            submod = converted_model.get_submodule(node.target)
            mod_type = type(submod).__name__
            modules[mod_type] = modules.get(mod_type, 0) + 1
        elif node.op == "call_function":
            fn_name = getattr(node.target, "__name__", str(node.target))
            functions[fn_name] = functions.get(fn_name, 0) + 1

    return {
        "modules": modules,
        "functions": functions,
    }


def quantize_model_ptq(
    model: nn.Module,
    model_architecture: str,
    calibration_dataloader: DataLoader,
    device: str = "cpu",
    example_input_shape: Tuple[int, ...] = (1, 3, 32, 32),
) -> Tuple[torch.jit.ScriptModule, Dict[str, Any]]:
    """
    Executes static Post-Training Quantization (PTQ) using PyTorch FX and QNNPACK backend.

    Parameters:
        model: Trained FP32 source PyTorch model.
        model_architecture: Architecture string ('resnet18_cifar' or 'mobilenet_v3_small_cifar').
        calibration_dataloader: DataLoader providing clean training samples for calibration.
        device: Execution device (must be 'cpu' for QNNPACK).
        example_input_shape: Input tensor shape for FX tracing.

    Returns:
        Tuple of (TorchScript traced INT8 model, operator inspection dictionary).
    """
    if device != "cpu":
        raise ValueError(f"QNNPACK INT8 quantization requires device='cpu', got '{device}'.")

    # 1. Ensure backend is configured
    torch.backends.quantized.engine = "qnnpack"

    # 2. Clone and set to eval mode
    float_model = copy.deepcopy(model).cpu().eval()

    # 3. Configure QNNPACK qconfig mapping
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")

    # 4. Architecture-specific handling:
    # MobileNetV3 InvertedResidual blocks produce activation scale ratios between residual branches
    # that can fall below QNNPACK's minimum threshold of 2^-14, causing runtime assert failures in
    # quantized add kernels. Leaving residual additions in FP32 ensures robust execution while
    # keeping all convolutional, linear, and SE layers in INT8.
    if "mobilenet" in model_architecture.lower():
        qconfig_mapping.set_object_type(torch.add, None)
        qconfig_mapping.set_object_type(operator.add, None)
        qconfig_mapping.set_object_type(operator.iadd, None)

    # 5. Prepare model with observers
    example_inputs = (torch.randn(*example_input_shape),)
    prepared_model = qfx.prepare_fx(float_model, qconfig_mapping, example_inputs)

    # 6. Calibration pass (strictly inference mode, no grad)
    with torch.inference_mode():
        for batch_inputs, _ in calibration_dataloader:
            prepared_model(batch_inputs.cpu())

    # 7. Convert model to quantized graph module
    converted_model = qfx.convert_fx(prepared_model)

    # 8. Inspect operator inventory
    operator_inventory = inspect_quantized_operators(converted_model)

    # 9. Trace into self-contained TorchScript module for standalone deployment
    traced_model = torch.jit.trace(converted_model, example_inputs)

    return traced_model, operator_inventory


def export_int8_checkpoint(
    traced_model: torch.jit.ScriptModule,
    model_architecture: str,
    output_path: Union[str, Path],
    source_checkpoint_path: Optional[Union[str, Path]] = None,
    source_checkpoint_sha256: Optional[str] = None,
    calibration_samples: int = 512,
    operator_inventory: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically saves a TorchScript INT8 model and companion metadata JSON file.

    Parameters:
        traced_model: Standalone TorchScript INT8 model.
        model_architecture: Architecture string.
        output_path: Target destination path (.pt).
        source_checkpoint_path: Path to source FP32 training checkpoint.
        source_checkpoint_sha256: SHA-256 of source checkpoint.
        calibration_samples: Number of calibration samples used.
        operator_inventory: Operator breakdown dictionary.
        extra_metadata: Optional additional metadata fields.

    Returns:
        Path to the saved .pt artifact.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_pt = dest.with_suffix(f".tmp.{os.getpid()}.pt")
    meta_path = dest.with_suffix(".json")
    temp_json = dest.with_suffix(f".tmp.{os.getpid()}.json")

    # Save TorchScript model atomically
    try:
        torch.jit.save(traced_model, str(temp_pt))
        os.replace(temp_pt, dest)
    except Exception:
        if temp_pt.exists():
            try:
                temp_pt.unlink()
            except OSError:
                pass
        raise

    # Save companion metadata JSON atomically
    metadata: Dict[str, Any] = {
        "model_architecture": model_architecture,
        "precision": "int8",
        "quantization_type": "post_training_static_quantization",
        "backend": "qnnpack",
        "num_classes": 10,
        "source_checkpoint_path": str(source_checkpoint_path) if source_checkpoint_path else None,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "calibration_samples": calibration_samples,
        "calibration_split": "train_partition_only",
        "calibration_seed": 42,
        "file_size_bytes": dest.stat().st_size,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "operator_inventory": operator_inventory or {},
    }
    if extra_metadata:
        metadata["extra_metadata"] = extra_metadata

    try:
        with open(temp_json, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(temp_json, meta_path)
    except Exception:
        if temp_json.exists():
            try:
                temp_json.unlink()
            except OSError:
                pass
        raise

    return dest

