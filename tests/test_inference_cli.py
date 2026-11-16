"""
Integration and end-to-end tests for inference CLI (infer.py) and benchmark runner (benchmark_inference.py).
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

PYTHON_EXE = sys.executable


@pytest.fixture
def sample_cifar_image(tmp_path: Path) -> Path:
    """Creates a temporary 32x32 RGB test image."""
    img_path = tmp_path / "sample_cifar.png"
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def sample_large_image(tmp_path: Path) -> Path:
    """Creates a temporary 64x64 RGB test image (requires resizing)."""
    img_path = tmp_path / "sample_64x64.png"
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


# =============================================================================
# 1. Single-Image Inference CLI Tests
# =============================================================================

def test_cli_single_image_success(sample_cifar_image: Path, tmp_path: Path):
    """Verify single-image inference runs with human output and saves valid JSON."""
    out_json = tmp_path / "single_out.json"
    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_cifar_image),
        "--model", "resnet18_cifar",
        "--precision", "fp32",
        "--device", "cpu",
        "--top-k", "5",
        "--output", str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"infer.py failed:\n{res.stderr}"
    assert "Edge Vision Reliability Lab — Inference Result" in res.stdout
    assert "Predicted Class:" in res.stdout

    # Verify JSON output
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "success"
    assert "predicted_class" in data["prediction"]
    assert len(data["prediction"]["top_k"]) == 5
    assert "preprocess_ms" in data["timing_ms"]
    assert "model_name" in data["metadata"]


def test_cli_single_image_json_stdout(sample_cifar_image: Path):
    """Verify --json routes valid machine-readable JSON to stdout and logs to stderr."""
    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_cifar_image),
        "--model", "resnet18_cifar",
        "--precision", "fp32",
        "--device", "cpu",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"infer.py failed:\n{res.stderr}"

    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert "sample_cifar" in res.stderr  # Diagnostic message routed to stderr


def test_cli_missing_image_path():
    """Verify nonexistent image path exits with returncode 1 without traceback."""
    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", "nonexistent/path/to/missing_img.png",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error: Image file does not exist" in res.stderr
    assert "Traceback" not in res.stderr


def test_cli_unsupported_image_format(tmp_path: Path):
    """Verify non-image text file triggers clean error and nonzero exit code."""
    bad_file = tmp_path / "corrupt_data.txt"
    bad_file.write_text("not a valid image format", encoding="utf-8")

    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(bad_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error: Failed to open or decode image" in res.stderr
    assert "Traceback" not in res.stderr


def test_cli_invalid_model_and_precision(sample_cifar_image: Path):
    """Verify unrecognized architecture or precision names exit cleanly."""
    cmd_bad_model = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_cifar_image),
        "--model", "vgg16_unknown",
    ]
    res_m = subprocess.run(cmd_bad_model, capture_output=True, text=True)
    assert res_m.returncode == 1
    assert "Error initializing model: Unsupported model architecture" in res_m.stderr


def test_cli_unsupported_device_combination(sample_cifar_image: Path):
    """Verify requesting INT8 on MPS exits cleanly with DeviceCompatibilityError message."""
    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_cifar_image),
        "--precision", "int8",
        "--device", "mps",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error initializing model: INT8 quantization backend" in res.stderr
    assert "Traceback" not in res.stderr


def test_cli_top_k_validation(sample_cifar_image: Path):
    """Verify top_k bounds validation in CLI."""
    for bad_k in ["0", "15"]:
        cmd = [
            PYTHON_EXE,
            "scripts/infer.py",
            "--image", str(sample_cifar_image),
            "--top-k", bad_k,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Error: top_k must be an integer between 1 and 10" in res.stderr


def test_cli_resize_flag_behavior(sample_large_image: Path):
    """Verify non-32x32 images are rejected without --resize, and succeed with --resize."""
    # 1. Without --resize -> rejection
    cmd_no_resize = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_large_image),
        "--model", "resnet18_cifar",
        "--device", "cpu",
    ]
    res_no = subprocess.run(cmd_no_resize, capture_output=True, text=True)
    assert res_no.returncode == 1
    assert "Set resize=True to enable explicit bilinear resizing" in res_no.stderr

    # 2. With --resize -> success
    cmd_resize = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--image", str(sample_large_image),
        "--model", "resnet18_cifar",
        "--device", "cpu",
        "--resize",
    ]
    res_yes = subprocess.run(cmd_resize, capture_output=True, text=True)
    assert res_yes.returncode == 0
    assert "Predicted Class:" in res_yes.stdout


# =============================================================================
# 2. Directory / Batch Inference CLI Tests
# =============================================================================

def test_cli_directory_inference(tmp_path: Path):
    """Verify batch directory inference processes images in deterministic sorted order."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 3 image files with names out of order
    for name in ["c_img.png", "a_img.png", "b_img.png"]:
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / name)

    out_json = tmp_path / "dir_out.json"
    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--dir", str(img_dir),
        "--model", "resnet18_cifar",
        "--precision", "fp32",
        "--device", "cpu",
        "--batch-size", "2",
        "--output", str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"infer.py failed:\n{res.stderr}"

    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_images"] == 3
    assert data["processed_successfully"] == 3
    assert data["errors_encountered"] == 0

    # Verify deterministic lexical ordering (a_img, b_img, c_img)
    file_names = [r["file_name"] for r in data["results"]]
    assert file_names == ["a_img.png", "b_img.png", "c_img.png"]


def test_cli_directory_unreadable_image_handling(tmp_path: Path):
    """Verify directory processing with --on-error record and --on-error stop."""
    img_dir = tmp_path / "mixed_images"
    img_dir.mkdir()

    # 1 valid image and 1 corrupted text file masquerading as image
    valid_arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(valid_arr).save(img_dir / "valid.png")
    (img_dir / "corrupted.png").write_text("corrupted content", encoding="utf-8")

    # Policy 1: --on-error record -> completes with partial success and records error
    out_json = tmp_path / "mixed_record.json"
    cmd_record = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--dir", str(img_dir),
        "--model", "resnet18_cifar",
        "--device", "cpu",
        "--on-error", "record",
        "--output", str(out_json),
    ]
    res_rec = subprocess.run(cmd_record, capture_output=True, text=True)
    assert res_rec.returncode == 0

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_images"] == 2
    assert data["processed_successfully"] == 1
    assert data["errors_encountered"] == 1

    corrupted_entry = next(r for r in data["results"] if r["file_name"] == "corrupted.png")
    assert corrupted_entry["status"] == "error"
    assert "error" in corrupted_entry

    # Policy 2: --on-error stop -> aborts immediately with returncode 1
    cmd_stop = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--dir", str(img_dir),
        "--model", "resnet18_cifar",
        "--device", "cpu",
        "--on-error", "stop",
    ]
    res_stop = subprocess.run(cmd_stop, capture_output=True, text=True)
    assert res_stop.returncode == 1
    assert "Aborting on unreadable image" in res_stop.stderr


def test_cli_directory_empty_rejection(tmp_path: Path):
    """Verify empty directory input raises clear error without traceback."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    cmd = [
        PYTHON_EXE,
        "scripts/infer.py",
        "--dir", str(empty_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "No supported image files" in res.stderr
    assert "Traceback" not in res.stderr


# =============================================================================
# 3. Benchmark Runner Tests
# =============================================================================

def test_benchmark_cli_smoke_test(tmp_path: Path):
    """Verify benchmark_inference.py runs a lightweight smoke test and creates valid CSV/JSON."""
    bench_dir = tmp_path / "bench_run"
    cmd = [
        PYTHON_EXE,
        "scripts/benchmark_inference.py",
        "--smoke-test",
        "--models", "resnet18_cifar",
        "--precisions", "fp32",
        "--devices", "cpu",
        "--output-dir", str(bench_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"benchmark_inference.py failed:\n{res.stderr}"

    csv_file = bench_dir / "benchmark_summary.csv"
    json_file = bench_dir / "benchmark_results.json"
    assert csv_file.is_file()
    assert json_file.is_file()

    # Verify CSV format
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2  # batch sizes 1 and 8
    assert rows[0]["model"] == "resnet18_cifar"
    assert float(rows[0]["throughput_fps"]) > 0.0
    assert float(rows[0]["median_total_ms"]) > 0.0

    # Verify JSON format & metadata completeness
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data["benchmark_metadata"]
    assert "timestamp_utc" in meta
    assert "git_commit" in meta
    assert "torch_version" in meta
    assert "tested_batch_sizes" in meta
    assert len(data["summary_table"]) == 2
    assert len(data["detailed_runs"]) == 1


def test_benchmark_cli_collision_protection(tmp_path: Path):
    """Verify benchmark runner protects existing output directories unless --overwrite is passed."""
    bench_dir = tmp_path / "collision_dir"
    cmd_base = [
        PYTHON_EXE,
        "scripts/benchmark_inference.py",
        "--smoke-test",
        "--models", "resnet18_cifar",
        "--precisions", "fp32",
        "--devices", "cpu",
        "--output-dir", str(bench_dir),
    ]

    # First run succeeds
    res1 = subprocess.run(cmd_base, capture_output=True, text=True)
    assert res1.returncode == 0

    # Second run without --overwrite fails with clear error
    res2 = subprocess.run(cmd_base, capture_output=True, text=True)
    assert res2.returncode == 1
    assert "Error: Output directory already contains benchmark artifacts" in res2.stderr

    # Third run with --overwrite succeeds
    res3 = subprocess.run(cmd_base + ["--overwrite"], capture_output=True, text=True)
    assert res3.returncode == 0


def test_benchmark_cli_invalid_filter_rejection():
    """Verify filtering for nonexistent model architectures fails cleanly."""
    cmd = [
        PYTHON_EXE,
        "scripts/benchmark_inference.py",
        "--models", "nonexistent_model",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error: No valid benchmark configurations match the specified filters" in res.stderr
