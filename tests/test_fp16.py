"""
Unit and integration tests for FP16 model conversion, checkpoint loading/export,
numerical stability, and evaluation infrastructure.
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_vision.evaluation.evaluator import (
    convert_model_to_fp16,
    export_inference_checkpoint,
    evaluate_model,
    load_checkpoint_for_evaluation,
)
from edge_vision.evaluation.robustness import RobustnessCondition, RobustnessEvaluator
from edge_vision.models import get_model, resnet18_cifar, mobilenet_v3_small_cifar


@pytest.mark.parametrize("model_name", ["resnet18_cifar", "mobilenet_v3_small_cifar"])
def test_convert_model_to_fp16_properties(model_name: str) -> None:
    """Verify that convert_model_to_fp16 transforms all parameters, buffers, and sets eval mode."""
    model = get_model(model_name, num_classes=10)
    model.train()  # Start in training mode with gradients enabled

    fp16_model = convert_model_to_fp16(model)

    assert fp16_model is model
    assert not fp16_model.training, "Model must be in eval mode after conversion"

    # All floating point parameters must be float16 and frozen
    for name, param in fp16_model.named_parameters():
        if param.is_floating_point():
            assert param.dtype == torch.float16, f"Parameter {name} is {param.dtype}, expected float16"
        assert not param.requires_grad, f"Parameter {name} still requires grad"

    # Buffers (running means/vars etc.) must also be float16 if floating point
    for name, buf in fp16_model.named_buffers():
        if buf.is_floating_point():
            assert buf.dtype == torch.float16, f"Buffer {name} is {buf.dtype}, expected float16"

    # Forward pass on float16 synthetic tensor
    dummy_input = torch.randn(2, 3, 32, 32, dtype=torch.float16)
    with torch.inference_mode():
        output = fp16_model(dummy_input)

    assert output.shape == (2, 10)
    assert output.dtype == torch.float16
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_load_checkpoint_precision_options() -> None:
    """Verify load_checkpoint_for_evaluation handles 'fp32', 'fp16', 'auto', and invalid strings."""
    model = resnet18_cifar(num_classes=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        torch.save(
            {
                "model_architecture": "resnet18_cifar",
                "model_state_dict": model.state_dict(),
                "num_classes": 10,
                "precision": "fp32",
            },
            ckpt_path,
        )

        # 1. Load as FP32
        m_fp32, meta_fp32 = load_checkpoint_for_evaluation(ckpt_path, precision="fp32", device="cpu")
        assert next(m_fp32.parameters()).dtype == torch.float32
        assert not m_fp32.training

        # 2. Load as FP16
        m_fp16, meta_fp16 = load_checkpoint_for_evaluation(ckpt_path, precision="fp16", device="cpu")
        assert next(m_fp16.parameters()).dtype == torch.float16
        assert not m_fp16.training

        # 3. Load as auto (should detect fp32)
        m_auto, meta_auto = load_checkpoint_for_evaluation(ckpt_path, precision="auto", device="cpu")
        assert meta_auto.get("precision", "fp32") == "fp32"
        assert next(m_auto.parameters()).dtype == torch.float32

        # 4. Invalid precision raises ValueError
        with pytest.raises(ValueError, match="Unsupported precision 'int4'"):
            load_checkpoint_for_evaluation(ckpt_path, precision="int4", device="cpu")


def test_export_inference_checkpoint_and_reload() -> None:
    """Verify export_inference_checkpoint creates valid artifacts and handles size reduction."""
    model_fp32 = mobilenet_v3_small_cifar(num_classes=10)
    model_fp16 = convert_model_to_fp16(mobilenet_v3_small_cifar(num_classes=10))

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = Path(tmpdir) / "model_fp32.pt"
        fp16_path = Path(tmpdir) / "model_fp16.pt"

        export_inference_checkpoint(
            model=model_fp32,
            model_architecture="mobilenet_v3_small_cifar",
            output_path=fp32_path,
            precision="fp32",
        )

        export_inference_checkpoint(
            model=model_fp16,
            model_architecture="mobilenet_v3_small_cifar",
            output_path=fp16_path,
            precision="fp16",
        )

        assert fp32_path.exists()
        assert fp16_path.exists()

        fp32_size = fp32_path.stat().st_size
        fp16_size = fp16_path.stat().st_size

        # FP16 checkpoint must be ~50% the size of FP32 (allowing small metadata overhead)
        ratio = fp32_size / fp16_size
        assert 1.9 < ratio < 2.1, f"Compression ratio {ratio:.3f} unexpected"

        # Reload both using auto
        loaded_fp32, meta32 = load_checkpoint_for_evaluation(fp32_path, precision="auto", device="cpu")
        assert meta32["precision"] == "fp32"
        assert next(loaded_fp32.parameters()).dtype == torch.float32

        loaded_fp16, meta16 = load_checkpoint_for_evaluation(fp16_path, precision="auto", device="cpu")
        assert meta16["precision"] == "fp16"
        assert next(loaded_fp16.parameters()).dtype == torch.float16


def test_synthetic_numerical_stability_and_concordance() -> None:
    """Verify output logits between FP32 and converted FP16 model have low absolute difference."""
    torch.manual_seed(42)
    model_fp32 = resnet18_cifar(num_classes=10).eval()

    # Create identical weights for FP16 model
    model_fp16 = resnet18_cifar(num_classes=10)
    model_fp16.load_state_dict(model_fp32.state_dict())
    convert_model_to_fp16(model_fp16)

    # Test batches
    for b_size in [1, 8, 32]:
        x = torch.randn(b_size, 3, 32, 32)
        with torch.inference_mode():
            out_fp32 = model_fp32(x)
            out_fp16 = model_fp16(x.half())

        assert not torch.isnan(out_fp16).any()
        assert not torch.isinf(out_fp16).any()

        # Check logit difference
        abs_diff = torch.abs(out_fp32 - out_fp16.float())
        mae = abs_diff.mean().item()
        max_err = abs_diff.max().item()

        # MAE and max error should be very small
        assert mae < 0.05, f"MAE {mae:.6f} exceeds threshold for batch size {b_size}"
        assert max_err < 0.15, f"Max error {max_err:.6f} exceeds threshold for batch size {b_size}"

        # Class predictions should match
        preds_32 = torch.argmax(out_fp32, dim=1)
        preds_16 = torch.argmax(out_fp16, dim=1)
        assert torch.equal(preds_32, preds_16), "Predictions diverged on synthetic input"


def test_robustness_evaluator_handles_fp16_model() -> None:
    """Verify RobustnessEvaluator automatically casts input batches to FP16 and computes loss in float32."""
    model = resnet18_cifar(num_classes=10)
    convert_model_to_fp16(model)

    # Dataloader produces float32 images
    images = torch.randn(16, 3, 32, 32, dtype=torch.float32)
    targets = torch.randint(0, 10, (16,), dtype=torch.long)
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    evaluator = RobustnessEvaluator(
        models={"resnet18": model},
        device=torch.device("cpu"),
        class_names=[f"c{i}" for i in range(10)],
    )
    cond = RobustnessCondition(condition_id="clean", is_clean=True, description="test")
    results = evaluator.evaluate_shared_batch_dataloader(
        dataloader=loader,
        condition=cond,
        clean_accuracies={"resnet18": 1.0},
    )

    assert "resnet18" in results
    res = results["resnet18"]
    assert res.metrics.total_samples == 16
    assert 0.0 <= res.metrics.accuracy <= 1.0
    assert not torch.isnan(torch.tensor(res.metrics.loss))
    assert not torch.isinf(torch.tensor(res.metrics.loss))
    assert res.condition_wall_time_seconds >= 0.0


def test_evaluate_model_precision_handling() -> None:
    """Verify evaluate_model handles an FP16 model with a float32 DataLoader."""
    model = mobilenet_v3_small_cifar(num_classes=10)
    convert_model_to_fp16(model)

    images = torch.randn(8, 3, 32, 32, dtype=torch.float32)
    targets = torch.randint(0, 10, (8,), dtype=torch.long)
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        class_names=[f"c{i}" for i in range(10)],
    )

    assert metrics.total_samples == 8
    assert 0.0 <= metrics.accuracy <= 1.0
    assert not torch.isnan(torch.tensor(metrics.loss))


def test_baseline_fp32_artifacts_preserved() -> None:
    """Verify that original baseline FP32 checkpoints and experiment results exist, are unaltered, and match hashes."""
    import hashlib

    base_dir = Path(__file__).resolve().parent.parent

    # Baseline checkpoints
    res_ckpt = base_dir / "experiments" / "exp_baseline_resnet18_seed42" / "checkpoints" / "best_checkpoint.pt"
    mbn_ckpt = base_dir / "experiments" / "exp_baseline_mobilenet_v3_small_seed42" / "checkpoints" / "best_checkpoint.pt"
    assert res_ckpt.exists(), "Original ResNet18 baseline checkpoint missing"
    assert mbn_ckpt.exists(), "Original MobileNetV3 baseline checkpoint missing"

    def compute_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    # Exact expected SHA-256 hashes of original trained checkpoints
    expected_res_sha = "56a238e67ad34321895d88da43ea182141a0355bf15bee94ca2dc51ccdedabff"
    expected_mbn_sha = "7243571c4d769b0a489ae3fd7392f407b80609de9e3c81f60284e21afca581c7"

    assert compute_sha256(res_ckpt) == expected_res_sha, "ResNet18 baseline checkpoint was mutated!"
    assert compute_sha256(mbn_ckpt) == expected_mbn_sha, "MobileNetV3 baseline checkpoint was mutated!"

    # FP32 Robustness results in reports/experiment_results
    fp32_results_csv = base_dir / "reports" / "experiment_results" / "robustness_fp32_results.csv"
    fp32_summary_json = base_dir / "reports" / "experiment_results" / "robustness_fp32_summary.json"
    fp32_meta_json = base_dir / "reports" / "experiment_results" / "robustness_fp32_metadata.json"

    assert fp32_results_csv.exists(), "FP32 robustness results CSV missing"
    assert fp32_summary_json.exists(), "FP32 robustness summary JSON missing"
    assert fp32_meta_json.exists(), "FP32 robustness metadata JSON missing"
