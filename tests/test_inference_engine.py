"""
Unit and integration tests for ArtifactRegistry and unified InferenceEngine.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.inference import (
    ArtifactRegistry,
    DeviceCompatibilityError,
    InferenceEngine,
    InferenceError,
    InvalidInputError,
    ModelNotFoundError,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)

HAS_MPS = torch.backends.mps.is_available()


# =============================================================================
# 1. Artifact Registry Tests
# =============================================================================

@pytest.mark.parametrize("model_name", ["resnet18_cifar", "mobilenet_v3_small_cifar"])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_registry_resolution_all_six_artifacts(model_name: str, precision: str):
    """Verify registry resolves all 6 canonical artifacts with exact paths and sizes."""
    info = ArtifactRegistry.resolve(model_name, precision)
    assert info.model_name == model_name
    assert info.precision == precision
    assert info.artifact_path.is_file()
    assert info.file_size_bytes > 0
    assert info.file_size_bytes == info.artifact_path.stat().st_size

    if precision == "int8":
        assert info.companion_metadata is not None
        assert info.companion_metadata["backend"] == "qnnpack"
        assert info.companion_metadata["precision"] == "int8"
        assert info.companion_metadata["model_architecture"] == model_name


def test_registry_checkpoint_override():
    """Verify that specifying an explicit checkpoint override path is honored."""
    canonical = ArtifactRegistry.resolve("resnet18_cifar", "fp32")
    override = ArtifactRegistry.resolve(
        "resnet18_cifar",
        "fp32",
        checkpoint_path=canonical.artifact_path,
    )
    assert override.artifact_path == canonical.artifact_path
    assert override.file_size_bytes == canonical.file_size_bytes


def test_registry_missing_artifact_error(tmp_path: Path):
    """Verify that a nonexistent checkpoint path raises ModelNotFoundError."""
    missing_file = tmp_path / "nonexistent_model.pt"
    with pytest.raises(ModelNotFoundError, match="Model artifact not found"):
        ArtifactRegistry.resolve(
            "resnet18_cifar",
            "fp32",
            checkpoint_path=missing_file,
        )


def test_registry_unsupported_model_and_precision():
    """Verify unsupported architecture or precision raises specific exceptions."""
    with pytest.raises(UnsupportedModelError, match="Unsupported model architecture"):
        ArtifactRegistry.resolve("vgg16", "fp32")

    with pytest.raises(UnsupportedPrecisionError, match="Unsupported precision"):
        ArtifactRegistry.resolve("resnet18_cifar", "bfloat16")


# =============================================================================
# 2. Device Compatibility & Model Loading Tests
# =============================================================================

def test_load_fp32_cpu():
    """Verify loading FP32 models on CPU."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    meta = engine.get_metadata()
    assert meta["device"] == "cpu"
    assert meta["backend"] == "pytorch_eager"
    assert meta["precision"] == "fp32"


@pytest.mark.skipif(not HAS_MPS, reason="Apple Silicon MPS not available")
def test_load_fp32_mps():
    """Verify loading FP32 models on MPS."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="mps")
    meta = engine.get_metadata()
    assert meta["device"] == "mps"
    assert meta["backend"] == "pytorch_eager"


@pytest.mark.skipif(not HAS_MPS, reason="Apple Silicon MPS not available")
def test_load_fp16_mps():
    """Verify loading FP16 models on MPS."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp16", device="mps")
    meta = engine.get_metadata()
    assert meta["device"] == "mps"
    assert meta["precision"] == "fp16"
    assert meta["model_dtype"] == "float16"


def test_load_fp16_cpu_warning():
    """Verify loading FP16 models on CPU records a device warning."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp16", device="cpu")
    meta = engine.get_metadata()
    assert meta["device"] == "cpu"
    assert meta["precision"] == "fp16"
    assert "device_warning" in meta
    assert "hardware-accelerated" in meta["device_warning"]


def test_load_int8_cpu():
    """Verify loading INT8 TorchScript models on CPU."""
    engine = InferenceEngine.load("resnet18_cifar", precision="int8", device="cpu")
    meta = engine.get_metadata()
    assert meta["device"] == "cpu"
    assert meta["backend"] == "torchscript"
    assert meta["precision"] == "int8"
    assert meta["model_dtype"] == "qint8"
    assert "quantization_metadata" in meta


def test_int8_mps_rejection():
    """Verify that requesting INT8 on MPS raises DeviceCompatibilityError."""
    with pytest.raises(DeviceCompatibilityError, match="QNNPACK.*strictly supported on CPU"):
        InferenceEngine.load("resnet18_cifar", precision="int8", device="mps")


def test_invalid_device_name():
    """Verify that requesting an unknown device name raises DeviceCompatibilityError."""
    with pytest.raises(DeviceCompatibilityError, match="Unsupported device"):
        InferenceEngine.load("resnet18_cifar", precision="fp32", device="cuda:0")


# =============================================================================
# 3. Single-Image and Batch Inference Tests Across All 6 Models
# =============================================================================

@pytest.mark.parametrize("model_name", ["resnet18_cifar", "mobilenet_v3_small_cifar"])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_single_image_inference_all_six_variants(model_name: str, precision: str):
    """Verify single-image inference runs cleanly on all 6 canonical variants."""
    engine = InferenceEngine.load(model_name, precision=precision, device="auto")
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))

    out = engine.predict_image(img, top_k=5)
    pred = out.prediction
    timing = out.timing

    # 1. Output types and dimensions
    assert pred.predicted_class in CIFAR10_CLASSES
    assert 0 <= pred.class_index < 10
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 10
    assert len(pred.logits) == 10
    assert len(pred.top_k) == 5

    # 2. Probability sums to ~1.0
    prob_sum = sum(pred.probabilities.values())
    assert pytest.approx(prob_sum, abs=1e-4) == 1.0

    # 3. Confidence consistency
    assert pred.confidence == pytest.approx(max(pred.probabilities.values()), abs=1e-6)
    assert pred.top_k[0] == (pred.predicted_class, pytest.approx(pred.confidence, abs=1e-6))

    # 4. Timing properties
    assert timing.preprocess_ms >= 0.0
    assert timing.forward_ms >= 0.0
    assert timing.postprocess_ms >= 0.0
    assert timing.total_ms >= 0.0
    assert timing.total_ms == pytest.approx(
        timing.preprocess_ms + timing.forward_ms + timing.postprocess_ms,
        abs=1e-3,
    )


@pytest.mark.parametrize("model_name", ["resnet18_cifar", "mobilenet_v3_small_cifar"])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_batch_inference_all_six_variants(model_name: str, precision: str):
    """Verify batch inference runs cleanly on all 6 canonical variants."""
    engine = InferenceEngine.load(model_name, precision=precision, device="auto")
    batch_arr = np.random.randint(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)

    out = engine.predict_batch(batch_arr, top_k=3)
    assert out.batch_size == 4
    assert len(out.predictions) == 4
    assert out.throughput_fps > 0.0

    for pred in out.predictions:
        assert pred.predicted_class in CIFAR10_CLASSES
        assert len(pred.top_k) == 3
        prob_sum = sum(pred.probabilities.values())
        assert pytest.approx(prob_sum, abs=1e-4) == 1.0


# =============================================================================
# 4. Top-K, Metadata, and Error Handling Tests
# =============================================================================

def test_top_k_ordering_and_validation():
    """Verify top-k predictions are strictly sorted descending and k is validated."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    raw = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    # Valid top_k = 10
    out = engine.predict_image(raw, top_k=10)
    top_k = out.prediction.top_k
    assert len(top_k) == 10
    probs = [p for _, p in top_k]
    assert probs == sorted(probs, reverse=True)

    # Invalid top_k = 0
    with pytest.raises(InvalidInputError, match="top_k must be an integer between 1 and 10"):
        engine.predict_image(raw, top_k=0)

    # Invalid top_k = 11
    with pytest.raises(InvalidInputError, match="top_k must be an integer between 1 and 10"):
        engine.predict_image(raw, top_k=11)


def test_metadata_completeness():
    """Verify get_metadata returns verified fields matching the model specification."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    meta = engine.get_metadata()
    assert meta["model_name"] == "resnet18_cifar"
    assert meta["precision"] == "fp32"
    assert meta["parameter_count"] == 11_173_962
    assert meta["artifact_size_bytes"] == 44_775_947
    assert meta["num_classes"] == 10
    assert meta["class_names"] == list(CIFAR10_CLASSES)
    assert meta["input_dimensions"] == [3, 32, 32]


def test_malformed_input_rejection():
    """Verify preprocessor errors are cleanly wrapped in InvalidInputError."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")

    # Wrong spatial dimensions with resize=False
    bad_img = Image.new("RGB", (64, 64))
    with pytest.raises(InvalidInputError, match="Image preprocessing failed"):
        engine.predict_image(bad_img, resize=False)

    # Malformed array with NaN
    nan_arr = np.zeros((32, 32, 3), dtype=np.float32)
    nan_arr[0, 0, 0] = np.nan
    with pytest.raises(InvalidInputError, match="NaN or infinite"):
        engine.predict_image(nan_arr)


# =============================================================================
# 5. Targeted Corrections & Regression Tests (Task 4.3 Audit)
# =============================================================================

@pytest.mark.parametrize("bad_k", [True, False, 0, -1, 11, 5.0, "5"])
def test_top_k_strict_rejection(bad_k):
    """Verify top_k strictly rejects bools, out-of-range ints, floats, and strings."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    raw = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    with pytest.raises(InvalidInputError, match="top_k must be an integer between 1 and 10"):
        engine.predict_image(raw, top_k=bad_k)


@pytest.mark.parametrize("valid_k", [1, 2, 5, 10])
def test_top_k_valid_integers(valid_k: int):
    """Verify valid top_k integer values return exact number of ranked predictions."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    raw = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    out = engine.predict_image(raw, top_k=valid_k)
    assert len(out.prediction.top_k) == valid_k


def test_tensor_input_predict_image_and_predict_batch():
    """Verify raw PyTorch tensors are accepted directly by predict_image and predict_batch."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")

    # Single CHW tensor (float in [0, 1])
    single_t = torch.rand(3, 32, 32, dtype=torch.float32)
    single_out = engine.predict_image(single_t, top_k=5)
    assert single_out.prediction.predicted_class in CIFAR10_CLASSES
    assert len(single_out.prediction.top_k) == 5

    # Batched 4D NCHW tensor
    batch_t = torch.rand(3, 3, 32, 32, dtype=torch.float32)
    batch_out = engine.predict_batch(batch_t, top_k=3)
    assert batch_out.batch_size == 3
    assert len(batch_out.predictions) == 3

    # Sequence of 3D CHW tensors
    seq_t = [torch.rand(3, 32, 32) for _ in range(2)]
    seq_out = engine.predict_batch(seq_t, top_k=3)
    assert seq_out.batch_size == 2


def test_empty_batch_predict_batch_rejection():
    """Verify calling predict_batch with an empty sequence raises InvalidInputError."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="cpu")
    with pytest.raises(InvalidInputError, match="Batch preprocessing failed: Empty batch provided"):
        engine.predict_batch([])


def test_incompatible_checkpoint_state_dict_error(tmp_path: Path):
    """Verify incompatible state-dict checkpoints raise InferenceError (not raw RuntimeError)."""
    bad_ckpt_path = tmp_path / "incompatible_resnet.pt"
    # Create checkpoint with invalid/mismatched tensor shapes
    torch.save(
        {"model_state_dict": {"conv1.weight": torch.randn(1, 1, 1, 1)}},
        bad_ckpt_path,
    )

    with pytest.raises(InferenceError, match="Incompatible state dict"):
        InferenceEngine.load(
            "resnet18_cifar",
            precision="fp32",
            checkpoint_path=bad_ckpt_path,
        )


def test_companion_json_syntax_error(tmp_path: Path):
    """Verify malformed JSON syntax in companion file raises InferenceError."""
    dummy_model = tmp_path / "dummy_model.pt"
    dummy_model.write_bytes(b"dummy")
    bad_json = tmp_path / "dummy_model.json"
    bad_json.write_text("{invalid_json_content", encoding="utf-8")

    with pytest.raises(InferenceError, match="Failed to read companion metadata JSON"):
        ArtifactRegistry.resolve(
            "resnet18_cifar",
            precision="fp32",
            checkpoint_path=dummy_model,
        )


def test_companion_json_metadata_mismatches(tmp_path: Path):
    """Verify architecture or precision conflicts in companion JSON raise InferenceError."""
    import json

    dummy_model = tmp_path / "dummy_model.pt"
    dummy_model.write_bytes(b"dummy")
    companion = tmp_path / "dummy_model.json"

    # 1. Architecture mismatch
    companion.write_text(
        json.dumps({"model_architecture": "mobilenet_v3_small_cifar", "precision": "fp32"}),
        encoding="utf-8",
    )
    with pytest.raises(InferenceError, match="Companion metadata architecture mismatch"):
        ArtifactRegistry.resolve(
            "resnet18_cifar",
            precision="fp32",
            checkpoint_path=dummy_model,
        )

    # 2. Precision mismatch
    companion.write_text(
        json.dumps({"model_architecture": "resnet18_cifar", "precision": "int8"}),
        encoding="utf-8",
    )
    with pytest.raises(InferenceError, match="Companion metadata precision mismatch"):
        ArtifactRegistry.resolve(
            "resnet18_cifar",
            precision="fp32",
            checkpoint_path=dummy_model,
        )


def test_companion_metadata_missing_policy(tmp_path: Path):
    """Verify that if companion JSON does not exist, ArtifactRegistry proceeds with None."""
    dummy_model = tmp_path / "dummy_no_json.pt"
    dummy_model.write_bytes(b"dummy")

    info = ArtifactRegistry.resolve(
        "resnet18_cifar",
        precision="fp32",
        checkpoint_path=dummy_model,
    )
    assert info.companion_metadata is None


@pytest.mark.skipif(not HAS_MPS, reason="Apple Silicon MPS not available")
def test_postprocessing_prediction_numerical_equivalence_mps():
    """Verify prediction values from optimized CPU postprocessing are numerically identical to GPU."""
    engine = InferenceEngine.load("resnet18_cifar", precision="fp32", device="mps")
    raw = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    # Output from optimized engine
    out = engine.predict_image(raw, top_k=5)

    # Raw model forward pass on MPS
    tensor = engine._preprocessor.preprocess_image(raw).unsqueeze(0)
    batch_tensor = engine._prepare_tensor_for_forward(tensor)
    with torch.inference_mode():
        logits_mps = engine._model(batch_tensor)
    torch.mps.synchronize()

    # Softmax directly on MPS
    probs_mps = torch.softmax(logits_mps.float(), dim=-1)

    # Compare values
    top_prob_mps = float(torch.max(probs_mps).item())
    top_idx_mps = int(torch.argmax(probs_mps).item())

    assert out.prediction.class_index == top_idx_mps
    assert out.prediction.predicted_class == CIFAR10_CLASSES[top_idx_mps]
    assert pytest.approx(out.prediction.confidence, abs=1e-6) == top_prob_mps
