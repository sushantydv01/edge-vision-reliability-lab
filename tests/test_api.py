"""
Integration and unit tests for the local REST inference API.

Tests all endpoints, model lifecycle management, input validation, bounded batching,
error handling, and concurrency safety.
"""

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from edge_vision.api.app import create_app, sanitize_message
from edge_vision.api.schemas import (
    BatchPredictionResponse,
    CurrentModelResponse,
    HealthResponse,
    ModelsListResponse,
    ModelLoadResponse,
    SinglePredictionResponse,
)
from edge_vision.api.service import (
    MAX_AGGREGATE_BATCH_BYTES,
    MAX_BATCH_SIZE,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_SIZE_BYTES,
    ModelManager,
    decode_image_bytes,
    read_bounded_upload,
)
from edge_vision.inference.exceptions import InvalidInputError, ModelNotFoundError


# =============================================================================
# Test Fixtures & Utilities
# =============================================================================

def make_test_image_bytes(
    size: Tuple[int, int] = (32, 32),
    color: Tuple[int, int, int] = (120, 180, 240),
    format: str = "PNG",
) -> bytes:
    """Creates in-memory image bytes with the specified dimensions and color."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.fixture
def api_client() -> TestClient:
    """Provides a TestClient with an isolated, uninitialized ModelManager."""
    app = create_app(model_manager=ModelManager())
    return TestClient(app)


@pytest.fixture
def loaded_api_client() -> TestClient:
    """Provides a TestClient with ResNet-18 FP32 CPU loaded."""
    manager = ModelManager()
    manager.load_model("resnet18_cifar", "fp32", device="cpu")
    app = create_app(model_manager=manager)
    return TestClient(app)


# =============================================================================
# 1. Health Endpoint Tests
# =============================================================================

def test_health_before_and_after_model_load(api_client: TestClient):
    """Verify /health reflects unloaded state initially, and loaded state after loading."""
    # 1. Before loading
    res_before = api_client.get("/health")
    assert res_before.status_code == 200
    data_before = res_before.json()
    assert data_before["status"] == "healthy"
    assert data_before["model_loaded"] is False
    assert data_before["model_name"] is None
    assert data_before["precision"] is None
    assert data_before["device"] is None
    HealthResponse(**data_before)

    # 2. Load model
    load_res = api_client.post(
        "/models/load",
        json={"model_name": "resnet18_cifar", "precision": "fp32", "device": "cpu"},
    )
    assert load_res.status_code == 200

    # 3. After loading
    res_after = api_client.get("/health")
    assert res_after.status_code == 200
    data_after = res_after.json()
    assert data_after["status"] == "healthy"
    assert data_after["model_loaded"] is True
    assert data_after["model_name"] == "resnet18_cifar"
    assert data_after["precision"] == "fp32"
    assert data_after["device"] == "cpu"
    assert data_after["backend"] == "pytorch_eager"
    HealthResponse(**data_after)


# =============================================================================
# 2. Model Registry & Inspection Tests
# =============================================================================

def test_get_models_list(api_client: TestClient):
    """Verify /models returns all 6 supported configurations from the registry."""
    res = api_client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 6
    assert len(data["models"]) == 6

    # Verify each model has expected schema
    parsed = ModelsListResponse(**data)
    configs = {(m.model_name, m.precision) for m in parsed.models}
    assert ("resnet18_cifar", "fp32") in configs
    assert ("resnet18_cifar", "fp16") in configs
    assert ("resnet18_cifar", "int8") in configs
    assert ("mobilenet_v3_small_cifar", "fp32") in configs
    assert ("mobilenet_v3_small_cifar", "fp16") in configs
    assert ("mobilenet_v3_small_cifar", "int8") in configs

    for m in parsed.models:
        assert m.parameter_count > 0
        assert m.file_size_bytes > 0
        assert len(m.supported_backends) > 0


def test_get_models_current(api_client: TestClient):
    """Verify /models/current returns unloaded and loaded representations."""
    # 1. Unloaded
    res = api_client.get("/models/current")
    assert res.status_code == 200
    unloaded_data = res.json()
    assert unloaded_data["loaded"] is False
    assert unloaded_data["model_name"] is None
    CurrentModelResponse(**unloaded_data)

    # 2. Load Mobilenet INT8
    api_client.post(
        "/models/load",
        json={"model_name": "mobilenet_v3_small_cifar", "precision": "int8", "device": "cpu"},
    )

    # 3. Loaded
    res_loaded = api_client.get("/models/current")
    assert res_loaded.status_code == 200
    loaded_data = res_loaded.json()
    assert loaded_data["loaded"] is True
    assert loaded_data["model_name"] == "mobilenet_v3_small_cifar"
    assert loaded_data["precision"] == "int8"
    assert loaded_data["backend"] == "torchscript"
    # Ensure no absolute file paths are exposed
    if loaded_data.get("metadata") and "artifact_path" in loaded_data["metadata"]:
        assert not loaded_data["metadata"]["artifact_path"].startswith("/")


# =============================================================================
# 3. Model Loading & Lifecycle Tests
# =============================================================================

def test_successful_model_loading(api_client: TestClient):
    """Verify POST /models/load initializes model and returns confirmation."""
    payload = {"model_name": "resnet18_cifar", "precision": "fp32", "device": "cpu"}
    res = api_client.post("/models/load", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["model_name"] == "resnet18_cifar"
    assert data["precision"] == "fp32"
    assert data["device"] == "cpu"
    assert data["load_time_ms"] > 0
    ModelLoadResponse(**data)


def test_unsupported_model_and_precision_errors(api_client: TestClient):
    """Verify loading unknown models or precisions triggers HTTP 400."""
    # Unsupported architecture
    res_bad_model = api_client.post(
        "/models/load",
        json={"model_name": "vgg16_cifar", "precision": "fp32"},
    )
    assert res_bad_model.status_code == 400
    assert res_bad_model.json()["error_type"] == "UnsupportedModelError"

    # Unsupported precision
    res_bad_prec = api_client.post(
        "/models/load",
        json={"model_name": "resnet18_cifar", "precision": "fp8"},
    )
    assert res_bad_prec.status_code == 400
    assert res_bad_prec.json()["error_type"] == "UnsupportedPrecisionError"


def test_device_compatibility_error(api_client: TestClient):
    """Verify requesting INT8 on MPS triggers clean HTTP 400 DeviceCompatibilityError."""
    res = api_client.post(
        "/models/load",
        json={"model_name": "resnet18_cifar", "precision": "int8", "device": "mps"},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error_type"] == "DeviceCompatibilityError"
    assert "does not support PyTorch quantized INT8" in data["detail"]


def test_missing_model_artifact_error(api_client: TestClient):
    """Verify missing artifact raises HTTP 404 ModelNotFoundError."""
    with patch(
        "edge_vision.inference.engine.ArtifactRegistry.resolve",
        side_effect=ModelNotFoundError("Model artifact not found for resnet18_cifar [fp32] at '/tmp/missing.pt'."),
    ):
        res = api_client.post(
            "/models/load",
            json={"model_name": "resnet18_cifar", "precision": "fp32"},
        )
        assert res.status_code == 404
        assert res.json()["error_type"] == "ModelNotFoundError"
        # Verify host path is sanitized
        assert "/tmp/" not in res.json()["detail"]


# =============================================================================
# 4. Single Image Prediction Tests
# =============================================================================

def test_predict_without_loaded_model_returns_409(api_client: TestClient):
    """Verify calling /predict when no model is resident returns HTTP 409 Conflict."""
    img_bytes = make_test_image_bytes()
    res = api_client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert res.status_code == 409
    assert "No model is currently loaded" in res.json()["detail"]


def test_predict_single_image_success(loaded_api_client: TestClient):
    """Verify standard single image inference returns complete predictions and timing."""
    img_bytes = make_test_image_bytes(size=(32, 32))
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("test_sample.png", img_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_sample.png"
    assert data["model_name"] == "resnet18_cifar"
    assert data["precision"] == "fp32"

    # Prediction details
    pred = data["prediction"]
    assert 0 <= pred["class_index"] <= 9
    assert isinstance(pred["predicted_class"], str)
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["top_k"]) == 5
    assert len(pred["probabilities"]) == 10
    assert len(pred["logits"]) == 10

    # Timing details
    timing = data["timing_ms"]
    assert timing["preprocess_ms"] >= 0.0
    assert timing["forward_ms"] >= 0.0
    assert timing["postprocess_ms"] >= 0.0
    assert timing["total_ms"] > 0.0

    SinglePredictionResponse(**data)


def test_predict_single_with_resize_behavior(loaded_api_client: TestClient):
    """Verify spatial dimensions: 64x64 rejected with resize=False, accepted with resize=True."""
    large_bytes = make_test_image_bytes(size=(64, 64))

    # 1. Default (resize=False) -> 400 Bad Request
    res_no_resize = loaded_api_client.post(
        "/predict",
        files={"file": ("large.png", large_bytes, "image/png")},
        params={"resize": False},
    )
    assert res_no_resize.status_code == 400
    assert "spatial dimensions 32x32" in res_no_resize.json()["detail"]

    # 2. Enabled (resize=True) -> 200 OK
    res_with_resize = loaded_api_client.post(
        "/predict",
        files={"file": ("large.png", large_bytes, "image/png")},
        params={"resize": True},
    )
    assert res_with_resize.status_code == 200
    assert res_with_resize.json()["status"] == "success"


def test_predict_top_k_parameter(loaded_api_client: TestClient):
    """Verify custom top_k values are respected."""
    img_bytes = make_test_image_bytes()
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("sample.png", img_bytes, "image/png")},
        params={"top_k": 3},
    )
    assert res.status_code == 200
    assert len(res.json()["prediction"]["top_k"]) == 3


def test_predict_invalid_top_k_values(loaded_api_client: TestClient):
    """Verify top_k < 1 or > 10 triggers validation error."""
    img_bytes = make_test_image_bytes()
    for bad_k in [0, 11, -5]:
        res = loaded_api_client.post(
            "/predict",
            files={"file": ("sample.png", img_bytes, "image/png")},
            params={"top_k": bad_k},
        )
        assert res.status_code in (400, 422)


# =============================================================================
# 5. Batch Prediction Tests
# =============================================================================

def test_predict_batch_success_and_ordering(loaded_api_client: TestClient):
    """Verify bounded batch inference preserves input order and returns throughput."""
    b1 = make_test_image_bytes(color=(255, 0, 0))
    b2 = make_test_image_bytes(color=(0, 255, 0))
    b3 = make_test_image_bytes(color=(0, 0, 255))

    files = [
        ("files", ("first_red.png", b1, "image/png")),
        ("files", ("second_green.png", b2, "image/png")),
        ("files", ("third_blue.png", b3, "image/png")),
    ]

    res = loaded_api_client.post("/predict/batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["batch_size"] == 3
    assert len(data["predictions"]) == 3
    assert data["throughput_fps"] > 0.0

    # Verify input order preservation
    assert data["predictions"][0]["filename"] == "first_red.png"
    assert data["predictions"][0]["index"] == 0
    assert data["predictions"][1]["filename"] == "second_green.png"
    assert data["predictions"][1]["index"] == 1
    assert data["predictions"][2]["filename"] == "third_blue.png"
    assert data["predictions"][2]["index"] == 2

    BatchPredictionResponse(**data)


def test_predict_batch_empty_list(loaded_api_client: TestClient):
    """Verify submitting empty batch returns HTTP 400 or 422."""
    res = loaded_api_client.post("/predict/batch", files=[])
    assert res.status_code in (400, 422)


def test_predict_batch_exceeds_maximum_ceiling(loaded_api_client: TestClient):
    """Verify exceeding MAX_BATCH_SIZE (64) is rejected with HTTP 400."""
    img_bytes = make_test_image_bytes()
    oversized_files = [
        ("files", (f"img_{i}.png", img_bytes, "image/png"))
        for i in range(MAX_BATCH_SIZE + 1)
    ]
    res = loaded_api_client.post("/predict/batch", files=oversized_files)
    assert res.status_code == 400
    assert "exceeds maximum allowed batch size" in res.json()["detail"]


# =============================================================================
# 6. Upload Validation & Security Edge Cases
# =============================================================================

def test_invalid_image_corrupted_bytes(loaded_api_client: TestClient):
    """Verify corrupted non-image bytes trigger HTTP 400."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["error_type"] == "InvalidInputError"
    assert "Failed to decode image" in res.json()["detail"]


def test_empty_upload_bytes(loaded_api_client: TestClient):
    """Verify empty 0-byte file upload triggers HTTP 400."""
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert res.status_code == 400
    assert "empty (0 bytes)" in res.json()["detail"]


def test_unsupported_file_extension(loaded_api_client: TestClient):
    """Verify non-image extensions are rejected."""
    txt_bytes = b"some text content"
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("document.txt", txt_bytes, "text/plain")},
    )
    assert res.status_code == 400
    assert "unsupported extension" in res.json()["detail"]


def test_oversized_upload_bytes(loaded_api_client: TestClient):
    """Verify uploads exceeding 10 MB ceiling are rejected with HTTP 400."""
    oversized_bytes = b"0" * (MAX_IMAGE_SIZE_BYTES + 1024)
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("huge.png", oversized_bytes, "image/png")},
    )
    assert res.status_code == 400
    assert "exceeds maximum allowed size" in res.json()["detail"]


def test_error_message_path_sanitization():
    """Verify internal filesystem paths are never leaked in error messages."""
    raw_error = "Checkpoint not found at /Users/sushantydv01/Desktop/edge-vision/checkpoints/best.pt"
    sanitized = sanitize_message(raw_error)
    assert "/Users/" not in sanitized
    assert "best.pt" in sanitized


# =============================================================================
# 7. Image Dimension Limits & Hardening Tests (Task 4.5.1)
# =============================================================================

def test_image_dimension_limits_width_exceeds_2048(loaded_api_client: TestClient):
    """Verify images with width > 2048 are rejected with HTTP 400."""
    wide_bytes = make_test_image_bytes(size=(2049, 32))
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("wide.png", wide_bytes, "image/png")},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error_type"] == "InvalidInputError"
    assert "dimensions (2049x32) exceed maximum allowed dimension of 2048" in data["detail"]


def test_image_dimension_limits_height_exceeds_2048(loaded_api_client: TestClient):
    """Verify images with height > 2048 are rejected with HTTP 400."""
    tall_bytes = make_test_image_bytes(size=(32, 2049))
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("tall.png", tall_bytes, "image/png")},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["error_type"] == "InvalidInputError"
    assert "dimensions (32x2049) exceed maximum allowed dimension of 2048" in data["detail"]


def test_image_dimension_limits_not_silently_resized_when_oversized(loaded_api_client: TestClient):
    """Verify oversized images (>2048) are rejected even when resize=True is requested."""
    wide_bytes = make_test_image_bytes(size=(2049, 32))
    res = loaded_api_client.post(
        "/predict?resize=true",
        files={"file": ("wide.png", wide_bytes, "image/png")},
    )
    assert res.status_code == 400
    assert "exceed maximum allowed dimension of 2048" in res.json()["detail"]


def test_image_dimension_limits_boundary_valid(loaded_api_client: TestClient):
    """Verify image with dimension exactly at 2048 boundary succeeds with resize=True."""
    boundary_bytes = make_test_image_bytes(size=(2048, 32))
    res = loaded_api_client.post(
        "/predict?resize=true",
        files={"file": ("boundary.png", boundary_bytes, "image/png")},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_image_dimension_limits_non_positive_dimensions():
    """Verify non-positive dimensions trigger InvalidInputError."""
    from unittest.mock import MagicMock

    mock_img_zero = MagicMock()
    mock_img_zero.size = (0, 32)
    mock_img_zero.__enter__.return_value = mock_img_zero

    with patch("PIL.Image.open", return_value=mock_img_zero):
        with pytest.raises(InvalidInputError) as exc_info:
            decode_image_bytes(b"dummy_data", filename="zero_width.png")
        assert "invalid non-positive dimensions" in str(exc_info.value)

    mock_img_neg = MagicMock()
    mock_img_neg.size = (-10, 32)
    mock_img_neg.__enter__.return_value = mock_img_neg

    with patch("PIL.Image.open", return_value=mock_img_neg):
        with pytest.raises(InvalidInputError) as exc_info:
            decode_image_bytes(b"dummy_data", filename="negative_width.png")
        assert "invalid non-positive dimensions" in str(exc_info.value)


def test_malformed_image_header_rejected(loaded_api_client: TestClient):
    """Verify malformed image header triggers HTTP 400 InvalidInputError."""
    malformed_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rBADCHUNK\xff\xff\xff\xff"
    res = loaded_api_client.post(
        "/predict",
        files={"file": ("malformed.png", malformed_header, "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["error_type"] == "InvalidInputError"
    assert "Failed to decode image" in res.json()["detail"]


def test_aggregate_batch_size_limit_exceeded(loaded_api_client: TestClient):
    """Verify exceeding aggregate batch limit returns HTTP 400."""
    b1 = make_test_image_bytes(size=(32, 32))
    b2 = make_test_image_bytes(size=(32, 32))

    files = [
        ("files", ("img1.png", b1, "image/png")),
        ("files", ("img2.png", b2, "image/png")),
    ]
    with patch("edge_vision.api.app.MAX_AGGREGATE_BATCH_BYTES", 100):
        res = loaded_api_client.post("/predict/batch", files=files)
        assert res.status_code == 400
        assert "exceeds maximum aggregate size limit" in res.json()["detail"]


def test_aggregate_batch_size_within_limit(loaded_api_client: TestClient):
    """Verify aggregate batch within limit succeeds."""
    b1 = make_test_image_bytes(size=(32, 32))
    b2 = make_test_image_bytes(size=(32, 32))

    files = [
        ("files", ("img1.png", b1, "image/png")),
        ("files", ("img2.png", b2, "image/png")),
    ]
    res = loaded_api_client.post("/predict/batch", files=files)
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["batch_size"] == 2


@pytest.mark.anyio
async def test_read_bounded_upload_stops_early():
    """Verify read_bounded_upload halts reading as soon as limit is exceeded."""
    from starlette.datastructures import UploadFile

    big_data = b"X" * (2 * 1024 * 1024)
    upload = UploadFile(io.BytesIO(big_data), filename="test_stream.bin")

    limit = 500 * 1024  # 500 KB
    chunk_size = 64 * 1024  # 64 KB

    with pytest.raises(InvalidInputError) as exc_info:
        await read_bounded_upload(
            upload,
            max_bytes=limit,
            chunk_size=chunk_size,
        )

    assert "exceeds maximum allowed size" in str(exc_info.value)
    # The underlying stream position should be at limit + chunk_size, NOT at 2 MB
    pos = upload.file.tell()
    assert pos <= limit + chunk_size
    assert pos < len(big_data)


# =============================================================================
# 7. Model Switching & Concurrency Safety Tests
# =============================================================================

def test_model_switching(api_client: TestClient):
    """Verify switching between ResNet FP32 and MobileNet INT8 executes cleanly."""
    img_bytes = make_test_image_bytes()

    # 1. Load ResNet-18 FP32
    res_load1 = api_client.post(
        "/models/load",
        json={"model_name": "resnet18_cifar", "precision": "fp32", "device": "cpu"},
    )
    assert res_load1.status_code == 200

    pred1 = api_client.post(
        "/predict",
        files={"file": ("img1.png", img_bytes, "image/png")},
    )
    assert pred1.status_code == 200
    assert pred1.json()["model_name"] == "resnet18_cifar"
    assert pred1.json()["precision"] == "fp32"

    # 2. Switch to MobileNetV3-Small INT8
    res_load2 = api_client.post(
        "/models/load",
        json={"model_name": "mobilenet_v3_small_cifar", "precision": "int8", "device": "cpu"},
    )
    assert res_load2.status_code == 200

    pred2 = api_client.post(
        "/predict",
        files={"file": ("img2.png", img_bytes, "image/png")},
    )
    assert pred2.status_code == 200
    assert pred2.json()["model_name"] == "mobilenet_v3_small_cifar"
    assert pred2.json()["precision"] == "int8"


def test_concurrent_predictions_and_model_switching(loaded_api_client: TestClient):
    """
    Verify concurrency safety: simultaneous predict and model-load operations
    do not deadlock, throw unhandled exceptions, or produce corrupted states.
    """
    img_bytes = make_test_image_bytes()

    def run_prediction():
        res = loaded_api_client.post(
            "/predict",
            files={"file": ("concurrent.png", img_bytes, "image/png")},
        )
        return res.status_code

    def run_switch():
        res = loaded_api_client.post(
            "/models/load",
            json={"model_name": "mobilenet_v3_small_cifar", "precision": "int8", "device": "cpu"},
        )
        return res.status_code

    with ThreadPoolExecutor(max_workers=6) as executor:
        # Launch 5 prediction tasks and 1 model switch task simultaneously
        predict_futures = [executor.submit(run_prediction) for _ in range(5)]
        switch_future = executor.submit(run_switch)

        switch_code = switch_future.result()
        predict_codes = [f.result() for f in predict_futures]

    assert switch_code == 200
    for code in predict_codes:
        assert code == 200
