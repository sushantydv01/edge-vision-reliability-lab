"""
Comprehensive unit tests for inference types and ImagePreprocessor pipeline.
"""

from typing import List
import numpy as np
import pytest
from PIL import Image
import torch

from edge_vision.data.transforms import get_cifar10_transforms
from edge_vision.inference import (
    PredictionResult,
    InferenceTiming,
    SingleImageInferenceOutput,
    BatchInferenceOutput,
    ImagePreprocessor,
)


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


# =============================================================================
# 1. Structured Inference Types Tests
# =============================================================================

def test_inference_types_construction():
    """Verify that structured inference dataclasses instantiate correctly and are immutable."""
    timing = InferenceTiming(
        preprocess_ms=0.5,
        forward_ms=3.2,
        postprocess_ms=0.1,
        total_ms=3.8,
    )
    assert timing.preprocess_ms == 0.5
    assert timing.total_ms == 3.8

    # Verify immutability
    with pytest.raises(Exception):
        timing.total_ms = 4.0  # dataclass(frozen=True)

    probs = {"airplane": 0.95, "automobile": 0.05}
    logits = [2.5, -0.5]
    pred = PredictionResult(
        predicted_class="airplane",
        class_index=0,
        confidence=0.95,
        probabilities=probs,
        logits=logits,
        top_k=[("airplane", 0.95), ("automobile", 0.05)],
    )
    assert pred.predicted_class == "airplane"
    assert pred.confidence == 0.95
    assert pred.top_k[0] == ("airplane", 0.95)

    single_out = SingleImageInferenceOutput(
        prediction=pred,
        timing=timing,
        metadata={"model": "resnet18_cifar", "precision": "fp32"},
    )
    assert single_out.metadata["model"] == "resnet18_cifar"

    batch_out = BatchInferenceOutput(
        predictions=[pred],
        timing=timing,
        batch_size=1,
        throughput_fps=263.15,
        metadata={"model": "resnet18_cifar"},
    )
    assert batch_out.batch_size == 1
    assert batch_out.throughput_fps == 263.15


# =============================================================================
# 2. PIL Image Input Tests
# =============================================================================

def test_pil_rgb_input(preprocessor):
    """Verify standard 32x32 PIL RGB image converts to a normalized (3, 32, 32) float32 tensor."""
    raw = np.full((32, 32, 3), 128, dtype=np.uint8)
    img = Image.fromarray(raw, mode="RGB")

    tensor = preprocessor.preprocess_image(img)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert tensor.device == torch.device("cpu")


def test_pil_grayscale_input(preprocessor):
    """Verify 32x32 PIL grayscale ('L') image expands to 3 RGB channels deterministically."""
    raw = np.full((32, 32), 200, dtype=np.uint8)
    img_gray = Image.fromarray(raw, mode="L")

    tensor = preprocessor.preprocess_image(img_gray)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    # All 3 normalized channels should be identical for a gray pixel (R == G == B before norm)
    # Note: after CIFAR normalization with different channel means/stds, channels differ
    # But unnormalized pixels were identical. We test that preprocessor accepts and converts 'L'.
    assert not torch.isnan(tensor).any()


def test_pil_rgba_input(preprocessor):
    """Verify 32x32 PIL RGBA image discards alpha channel without error."""
    raw_rgba = np.full((32, 32, 4), 100, dtype=np.uint8)
    raw_rgba[:, :, 3] = 255  # Alpha channel
    img_rgba = Image.fromarray(raw_rgba, mode="RGBA")

    tensor = preprocessor.preprocess_image(img_rgba)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


# =============================================================================
# 3. NumPy Array Input Tests
# =============================================================================

def test_numpy_uint8_input(preprocessor):
    """Verify 32x32x3 uint8 NumPy array in [0, 255] normalizes properly."""
    arr = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_image(arr)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()


def test_numpy_float32_input(preprocessor):
    """Verify 32x32x3 float32 NumPy array in [0.0, 1.0] normalizes properly."""
    arr = np.random.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)
    tensor = preprocessor.preprocess_image(arr)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_numpy_float64_input(preprocessor):
    """Verify 32x32x3 float64 NumPy array in [0.0, 1.0] converts to float32 tensor."""
    arr = np.random.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float64)
    tensor = preprocessor.preprocess_image(arr)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_numpy_grayscale_inputs(preprocessor):
    """Verify 2D (32, 32) and 3D (32, 32, 1) NumPy arrays expand to 3 channels."""
    arr_2d = np.full((32, 32), 128, dtype=np.uint8)
    t_2d = preprocessor.preprocess_image(arr_2d)
    assert t_2d.shape == (3, 32, 32)

    arr_3d = np.full((32, 32, 1), 128, dtype=np.uint8)
    t_3d = preprocessor.preprocess_image(arr_3d)
    assert t_3d.shape == (3, 32, 32)
    assert torch.equal(t_2d, t_3d)


def test_numpy_rgba_input(preprocessor):
    """Verify (32, 32, 4) NumPy array discards alpha channel."""
    arr_rgba = np.full((32, 32, 4), 150, dtype=np.uint8)
    arr_rgb = np.full((32, 32, 3), 150, dtype=np.uint8)

    t_rgba = preprocessor.preprocess_image(arr_rgba)
    t_rgb = preprocessor.preprocess_image(arr_rgb)
    assert t_rgba.shape == (3, 32, 32)
    assert torch.equal(t_rgba, t_rgb)


# =============================================================================
# 4. PyTorch Tensor Input Tests
# =============================================================================

def test_tensor_chw_float_input(preprocessor):
    """Verify (3, 32, 32) float32 tensor in [0.0, 1.0] normalizes correctly."""
    t = torch.rand(3, 32, 32, dtype=torch.float32)
    out = preprocessor.preprocess_image(t)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_tensor_chw_uint8_input(preprocessor):
    """Verify (3, 32, 32) uint8 tensor in [0, 255] scales and normalizes correctly."""
    t = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out = preprocessor.preprocess_image(t)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_tensor_grayscale_input(preprocessor):
    """Verify (1, 32, 32) tensor expands to 3 channels."""
    t_gray = torch.full((1, 32, 32), 0.5, dtype=torch.float32)
    out = preprocessor.preprocess_image(t_gray)
    assert out.shape == (3, 32, 32)


def test_tensor_rgba_input(preprocessor):
    """Verify (4, 32, 32) tensor discards alpha channel."""
    t_rgba = torch.full((4, 32, 32), 0.7, dtype=torch.float32)
    out = preprocessor.preprocess_image(t_rgba)
    assert out.shape == (3, 32, 32)


# =============================================================================
# 5. Batched Preprocessing Tests
# =============================================================================

def test_batch_nchw_tensor(preprocessor):
    """Verify 4D PyTorch tensor (4, 3, 32, 32) normalizes to (4, 3, 32, 32)."""
    batch_t = torch.rand(4, 3, 32, 32, dtype=torch.float32)
    out = preprocessor.preprocess_batch(batch_t)
    assert out.shape == (4, 3, 32, 32)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_batch_nhwc_numpy(preprocessor):
    """Verify 4D NumPy array (4, 32, 32, 3) normalizes to (4, 3, 32, 32)."""
    batch_arr = np.random.randint(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    out = preprocessor.preprocess_batch(batch_arr)
    assert out.shape == (4, 3, 32, 32)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_batch_list_mixed(preprocessor):
    """Verify sequence of mixed PIL, NumPy, and PyTorch inputs normalizes to (3, 3, 32, 32)."""
    raw_np = np.full((32, 32, 3), 100, dtype=np.uint8)
    pil_img = Image.fromarray(raw_np)
    tensor = torch.from_numpy(raw_np).permute(2, 0, 1).to(torch.float32) / 255.0

    batch = [pil_img, raw_np, tensor]
    out = preprocessor.preprocess_batch(batch)
    assert out.shape == (3, 3, 32, 32)
    assert out.dtype == torch.float32

    # All three inputs represent identical pixels -> outputs must be numerically identical
    assert torch.allclose(out[0], out[1], atol=1e-7)
    assert torch.allclose(out[0], out[2], atol=1e-7)


def test_empty_batch_rejection(preprocessor):
    """Verify passing an empty sequence raises ValueError."""
    with pytest.raises(ValueError, match="Empty batch provided"):
        preprocessor.preprocess_batch([])


# =============================================================================
# 6. Input Validation & Error Handling Tests
# =============================================================================

def test_invalid_dimensions(preprocessor):
    """Verify 1D, 5D, or invalid dimension arrays/tensors raise ValueError."""
    # 1D array
    with pytest.raises(ValueError, match="Unsupported NumPy array dimensions"):
        preprocessor.preprocess_image(np.array([1, 2, 3], dtype=np.uint8))

    # 5D tensor
    with pytest.raises(ValueError, match="Unsupported PyTorch tensor dimensions"):
        preprocessor.preprocess_image(torch.zeros(1, 1, 3, 32, 32))


def test_invalid_channel_counts(preprocessor):
    """Verify 2-channel or 5-channel inputs raise ValueError."""
    # 2-channel NumPy
    with pytest.raises(ValueError, match="Expected 1, 3, or 4 channels"):
        preprocessor.preprocess_image(np.zeros((32, 32, 2), dtype=np.uint8))

    # 5-channel Tensor
    with pytest.raises(ValueError, match="Expected PyTorch tensor with 1, 3, or 4 channels"):
        preprocessor.preprocess_image(torch.zeros(5, 32, 32))


def test_ambiguous_layout_rejection(preprocessor):
    """Verify ambiguous CHW NumPy or HWC PyTorch tensors are rejected rather than silently transposed."""
    # NumPy passed as (3, 32, 32) instead of (32, 32, 3)
    with pytest.raises(ValueError, match="Ambiguous NumPy array layout"):
        preprocessor.preprocess_image(np.zeros((3, 32, 32), dtype=np.uint8))

    # PyTorch Tensor passed as (32, 32, 3) instead of (3, 32, 32)
    with pytest.raises(ValueError, match="Ambiguous PyTorch tensor layout"):
        preprocessor.preprocess_image(torch.zeros(32, 32, 3))


def test_invalid_dtypes(preprocessor):
    """Verify non-supported dtypes (int32, bool, etc.) raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported NumPy array dtype"):
        preprocessor.preprocess_image(np.zeros((32, 32, 3), dtype=np.int32))

    with pytest.raises(TypeError, match="Unsupported PyTorch tensor dtype"):
        preprocessor.preprocess_image(torch.zeros(3, 32, 32, dtype=torch.int64))


def test_nan_and_infinity_rejection(preprocessor):
    """Verify arrays or tensors containing NaN or Inf raise ValueError."""
    # NumPy NaN
    bad_arr = np.full((32, 32, 3), 0.5, dtype=np.float32)
    bad_arr[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        preprocessor.preprocess_image(bad_arr)

    # Tensor Inf
    bad_t = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
    bad_t[1, 5, 5] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        preprocessor.preprocess_image(bad_t)


def test_invalid_value_ranges(preprocessor):
    """Verify floating-point inputs outside [0.0, 1.0] raise ValueError."""
    # Negative float
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        preprocessor.preprocess_image(np.full((32, 32, 3), -0.1, dtype=np.float32))

    # Float > 1.0 (e.g. unscaled 0-255 float)
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        preprocessor.preprocess_image(torch.full((3, 32, 32), 255.0, dtype=torch.float32))


# =============================================================================
# 7. Resizing Behavior Tests
# =============================================================================

def test_resizing_disabled_rejects_non_32x32(preprocessor):
    """Verify that resize=False strictly rejects non-32x32 images."""
    # PIL 64x64
    img_64 = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="Expected PIL Image spatial dimensions 32x32"):
        preprocessor.preprocess_image(img_64, resize=False)

    # NumPy 48x48
    arr_48 = np.zeros((48, 48, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected NumPy array spatial dimensions 32x32"):
        preprocessor.preprocess_image(arr_48, resize=False)

    # Tensor 28x28
    t_28 = torch.zeros(3, 28, 28)
    with pytest.raises(ValueError, match="Expected PyTorch tensor spatial dimensions 32x32"):
        preprocessor.preprocess_image(t_28, resize=False)


def test_resizing_enabled(preprocessor):
    """Verify that resize=True deterministically scales images to 32x32."""
    img_64 = Image.new("RGB", (64, 48), color=(120, 140, 160))
    t_pil = preprocessor.preprocess_image(img_64, resize=True)
    assert t_pil.shape == (3, 32, 32)

    arr_64 = np.full((64, 48, 3), 120, dtype=np.uint8)
    t_np = preprocessor.preprocess_image(arr_64, resize=True)
    assert t_np.shape == (3, 32, 32)

    t_tensor = torch.full((3, 64, 48), 0.5, dtype=torch.float32)
    t_out = preprocessor.preprocess_image(t_tensor, resize=True)
    assert t_out.shape == (3, 32, 32)


# =============================================================================
# 8. Immutability & Determinism Tests
# =============================================================================

def test_input_immutability(preprocessor):
    """Verify caller-owned arrays and tensors are never modified in-place."""
    orig_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    np_copy = orig_np.copy()
    _ = preprocessor.preprocess_image(orig_np)
    assert np.array_equal(orig_np, np_copy), "NumPy array was mutated in-place by preprocessor!"

    orig_t = torch.rand(3, 32, 32, dtype=torch.float32)
    t_copy = orig_t.clone()
    _ = preprocessor.preprocess_image(orig_t)
    assert torch.equal(orig_t, t_copy), "PyTorch tensor was mutated in-place by preprocessor!"


def test_deterministic_repeated_preprocessing(preprocessor):
    """Verify repeated calls with the same input yield bit-for-bit identical tensors."""
    raw = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(raw)

    out1 = preprocessor.preprocess_image(img)
    out2 = preprocessor.preprocess_image(img)
    assert torch.equal(out1, out2)


# =============================================================================
# 9. Numerical Equivalence with Canonical CIFAR-10 Transforms
# =============================================================================

def test_numerical_parity_with_cifar10_transforms(preprocessor):
    """
    Verify exact numerical parity between ImagePreprocessor and the canonical
    evaluation transform (get_cifar10_transforms(train=False, augment=False))
    on native 32x32 RGB PIL images.
    """
    canonical_pipeline = get_cifar10_transforms(train=False, augment=False)

    rng = np.random.RandomState(42)
    for _ in range(5):
        raw_pixels = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        pil_image = Image.fromarray(raw_pixels, mode="RGB")

        expected_tensor = canonical_pipeline(pil_image)
        actual_tensor = preprocessor.preprocess_image(pil_image)

        max_diff = torch.max(torch.abs(expected_tensor - actual_tensor)).item()
        assert max_diff < 1e-7, f"Parity mismatch: max difference {max_diff} exceeds 1e-7!"
        assert torch.allclose(expected_tensor, actual_tensor, atol=1e-7)
