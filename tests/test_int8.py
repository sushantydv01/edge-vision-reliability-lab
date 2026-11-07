"""
Unit and integration tests for INT8 Post-Training Static Quantization (PTQ) pipeline.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_vision.data.config import load_data_config
from edge_vision.data.dataset import get_or_create_split_indices
from edge_vision.evaluation.evaluator import evaluate_model, load_checkpoint_for_evaluation
from edge_vision.evaluation.robustness import RobustnessCondition, RobustnessEvaluator
from edge_vision.models import resnet18_cifar, mobilenet_v3_small_cifar
from edge_vision.quantization.ptq import (
    get_calibration_dataloader,
    quantize_model_ptq,
    export_int8_checkpoint,
    inspect_quantized_operators,
)
from torchvision import datasets


def test_calibration_split_isolation() -> None:
    """Verify calibration samples strictly come from the train split and never val or test."""
    cfg = load_data_config("configs/cifar10.yaml")
    raw_train = datasets.CIFAR10(root=cfg.dataset.raw_dir, train=True, download=False, transform=None)
    train_idx, val_idx = get_or_create_split_indices(
        manifest_path=cfg.split.manifest_path,
        total_samples=len(raw_train),
        train_size=cfg.split.train_size,
        val_size=cfg.split.val_size,
        seed=cfg.split.seed,
    )

    train_set = set(train_idx)
    val_set = set(val_idx)

    # Acquire calibration DataLoader
    calib_loader = get_calibration_dataloader(
        data_config_path="configs/cifar10.yaml",
        num_samples=256,
        batch_size=64,
        seed=42,
    )

    calib_indices = calib_loader.dataset.indices
    assert len(calib_indices) == 256
    # Must be subset of train_set
    assert set(calib_indices).issubset(train_set), "Calibration indices contain non-train samples!"
    # Zero overlap with val_set
    assert len(set(calib_indices).intersection(val_set)) == 0, "Calibration indices overlap with validation partition!"


def test_calibration_determinism() -> None:
    """Verify calibration DataLoader produces identical samples across repeated calls with same seed."""
    loader1 = get_calibration_dataloader(num_samples=128, seed=42)
    loader2 = get_calibration_dataloader(num_samples=128, seed=42)
    loader3 = get_calibration_dataloader(num_samples=128, seed=99)

    assert loader1.dataset.indices == loader2.dataset.indices
    assert loader1.dataset.indices != loader3.dataset.indices


def test_resnet18_ptq_conversion_and_finite_outputs() -> None:
    """Verify ResNet18 converts to INT8 on CPU and produces valid, finite logits."""
    model = resnet18_cifar(num_classes=10).eval()
    calib_loader = get_calibration_dataloader(num_samples=64, batch_size=32, seed=42)

    traced_model, op_inv = quantize_model_ptq(
        model=model,
        model_architecture="resnet18_cifar",
        calibration_dataloader=calib_loader,
        device="cpu",
    )

    assert isinstance(traced_model, torch.jit.ScriptModule)
    # Test forward pass with batch size 1 and 4
    for b in [1, 4]:
        x = torch.randn(b, 3, 32, 32)
        out = traced_model(x)
        assert out.shape == (b, 10)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # Verify quantized operators exist
    modules = op_inv["modules"]
    assert "Conv2d" in modules or "ConvReLU2d" in modules
    assert "Linear" in modules


def test_mobilenet_v3_small_residual_handling() -> None:
    """Verify MobileNetV3-Small converts without QNNPACK 2^-14 scale ratio crash."""
    model = mobilenet_v3_small_cifar(num_classes=10).eval()
    calib_loader = get_calibration_dataloader(num_samples=64, batch_size=32, seed=42)

    traced_model, op_inv = quantize_model_ptq(
        model=model,
        model_architecture="mobilenet_v3_small_cifar",
        calibration_dataloader=calib_loader,
        device="cpu",
    )

    assert isinstance(traced_model, torch.jit.ScriptModule)
    x = torch.randn(2, 3, 32, 32)
    out = traced_model(x)
    assert out.shape == (2, 10)
    assert not torch.isnan(out).any()

    # Operators check: Conv layers are quantized
    modules = op_inv["modules"]
    assert "Conv2d" in modules or "ConvReLU2d" in modules


def test_export_int8_checkpoint_and_independent_reload() -> None:
    """Verify export_int8_checkpoint writes atomic .pt and .json, and reloads without training source."""
    model = resnet18_cifar(num_classes=10).eval()
    calib_loader = get_calibration_dataloader(num_samples=64, batch_size=32, seed=42)

    traced_model, op_inv = quantize_model_ptq(
        model=model,
        model_architecture="resnet18_cifar",
        calibration_dataloader=calib_loader,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pt = Path(tmpdir) / "test_model_int8.pt"
        exported = export_int8_checkpoint(
            traced_model=traced_model,
            model_architecture="resnet18_cifar",
            output_path=out_pt,
            calibration_samples=64,
            operator_inventory=op_inv,
        )

        assert exported.exists()
        assert exported.stat().st_size > 0
        meta_json = exported.with_suffix(".json")
        assert meta_json.exists()

        # Reload independently
        reloaded, meta = load_checkpoint_for_evaluation(exported, device=torch.device("cpu"), precision="int8")
        assert meta["precision"] == "int8"
        x = torch.randn(2, 3, 32, 32)
        out = reloaded(x)
        assert out.shape == (2, 10)


def test_evaluator_handles_int8_model() -> None:
    """Verify RobustnessEvaluator and evaluate_model handle INT8 models without StopIteration."""
    model = resnet18_cifar(num_classes=10).eval()
    calib_loader = get_calibration_dataloader(num_samples=64, batch_size=32, seed=42)
    traced_model, _ = quantize_model_ptq(model, "resnet18_cifar", calib_loader, device="cpu")

    images = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=4)

    metrics = evaluate_model(traced_model, loader, criterion=nn.CrossEntropyLoss(), device=torch.device("cpu"))
    assert metrics.total_samples == 8
    assert 0.0 <= metrics.accuracy <= 1.0
    assert not torch.isnan(torch.tensor(metrics.loss))


def test_baseline_fp32_and_fp16_artifacts_preserved() -> None:
    """Verify that original baseline FP32 and FP16 artifacts remain unaltered."""
    import hashlib

    base_dir = Path(__file__).resolve().parent.parent

    # Baseline checkpoints
    res_ckpt = base_dir / "experiments" / "exp_baseline_resnet18_seed42" / "checkpoints" / "best_checkpoint.pt"
    mbn_ckpt = base_dir / "experiments" / "exp_baseline_mobilenet_v3_small_seed42" / "checkpoints" / "best_checkpoint.pt"

    def compute_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    expected_res_sha = "56a238e67ad34321895d88da43ea182141a0355bf15bee94ca2dc51ccdedabff"
    expected_mbn_sha = "7243571c4d769b0a489ae3fd7392f407b80609de9e3c81f60284e21afca581c7"

    assert compute_sha256(res_ckpt) == expected_res_sha
    assert compute_sha256(mbn_ckpt) == expected_mbn_sha

