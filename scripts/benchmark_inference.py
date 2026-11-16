#!/usr/bin/env python3
"""
Comprehensive Inference Latency and Throughput Benchmark Runner.

Benchmarks the six canonical model/precision variants across valid compute devices:
  - ResNet-18 (FP32, FP16, INT8)
  - MobileNetV3-Small (FP32, FP16, INT8)

Hardware & Execution Devices:
  - FP32: CPU and MPS (Apple Silicon GPU)
  - FP16: MPS (Apple Silicon GPU); CPU included as diagnostic (labelled)
  - INT8: CPU only (strictly enforced via QNNPACK backend)

Metrics Measured:
  - Model loading time (isolated from inference)
  - Latency breakdown: preprocessing, forward pass, postprocessing, end-to-end
  - Median, mean, std, min, max, p95 latency
  - Throughput (images/sec) across batch sizes 1, 8, 32, and 64
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision

from edge_vision.inference import (
    InferenceEngine,
    InferenceError,
    DeviceCompatibilityError,
    ModelNotFoundError,
    UnsupportedModelError,
    UnsupportedPrecisionError,
)
from edge_vision.inference.registry import ArtifactRegistry, find_project_root


BENCHMARK_CONFIGURATIONS = [
    # ResNet-18
    {"model": "resnet18_cifar", "precision": "fp32", "device": "cpu", "is_diagnostic": False, "note": "Canonical CPU FP32 baseline"},
    {"model": "resnet18_cifar", "precision": "fp32", "device": "mps", "is_diagnostic": False, "note": "Accelerated MPS FP32"},
    {"model": "resnet18_cifar", "precision": "fp16", "device": "mps", "is_diagnostic": False, "note": "Accelerated MPS FP16"},
    {"model": "resnet18_cifar", "precision": "fp16", "device": "cpu", "is_diagnostic": True, "note": "CPU FP16 diagnostic (lacks hardware vector GEMM)"},
    {"model": "resnet18_cifar", "precision": "int8", "device": "cpu", "is_diagnostic": False, "note": "Quantized INT8 CPU (QNNPACK)"},
    # MobileNetV3-Small
    {"model": "mobilenet_v3_small_cifar", "precision": "fp32", "device": "cpu", "is_diagnostic": False, "note": "Canonical CPU FP32 baseline"},
    {"model": "mobilenet_v3_small_cifar", "precision": "fp32", "device": "mps", "is_diagnostic": False, "note": "Accelerated MPS FP32"},
    {"model": "mobilenet_v3_small_cifar", "precision": "fp16", "device": "mps", "is_diagnostic": False, "note": "Accelerated MPS FP16"},
    {"model": "mobilenet_v3_small_cifar", "precision": "fp16", "device": "cpu", "is_diagnostic": True, "note": "CPU FP16 diagnostic (lacks hardware vector GEMM)"},
    {"model": "mobilenet_v3_small_cifar", "precision": "int8", "device": "cpu", "is_diagnostic": False, "note": "Quantized INT8 CPU (QNNPACK)"},
]


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """Computes summary statistics (mean, median, std, min, max, p95) for a series."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "std": round(float(np.std(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
    }


def run_benchmark_configuration(
    cfg: Dict[str, Any],
    batch_sizes: List[int],
    warmup: int,
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Executes benchmark measurements for a single model/precision/device configuration.
    """
    model_name = cfg["model"]
    precision = cfg["precision"]
    device = cfg["device"]

    # 1. Measure model loading time strictly isolated from inference latency
    t_load_start = time.perf_counter()
    engine = InferenceEngine.load(model_name=model_name, precision=precision, device=device)
    t_load_end = time.perf_counter()
    load_time_ms = round((t_load_end - t_load_start) * 1000.0, 3)

    meta = engine.get_metadata()
    artifact_path = Path(meta["artifact_path"])
    artifact_sha = compute_sha256(artifact_path)

    batch_results: Dict[int, Dict[str, Any]] = {}

    for b_sz in batch_sizes:
        # Generate deterministic synthetic batch input (NHWC uint8)
        rng = np.random.RandomState(seed + b_sz)
        raw_batch = rng.randint(0, 256, size=(b_sz, 32, 32, 3), dtype=np.uint8)

        # Warmup passes (unmeasured, excluded from statistics)
        for _ in range(warmup):
            _ = engine.predict_batch(raw_batch, resize=False)
            if device == "mps":
                torch.mps.synchronize()

        # Measured repetitions
        wall_latencies_ms: List[float] = []
        engine_total_ms: List[float] = []
        preprocess_ms: List[float] = []
        forward_ms: List[float] = []
        postprocess_ms: List[float] = []

        for _ in range(repetitions):
            if device == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            out = engine.predict_batch(raw_batch, resize=False)
            if device == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()

            wall_latencies_ms.append((t1 - t0) * 1000.0)
            engine_total_ms.append(out.timing.total_ms)
            preprocess_ms.append(out.timing.preprocess_ms)
            forward_ms.append(out.timing.forward_ms)
            postprocess_ms.append(out.timing.postprocess_ms)

        stats_wall = compute_summary_stats(wall_latencies_ms)
        stats_engine = compute_summary_stats(engine_total_ms)
        stats_forward = compute_summary_stats(forward_ms)
        stats_preprocess = compute_summary_stats(preprocess_ms)
        stats_postprocess = compute_summary_stats(postprocess_ms)

        # Calculate throughput using median wall-clock time
        throughput_fps = round((b_sz / (stats_wall["median"] / 1000.0)), 1) if stats_wall["median"] > 0 else 0.0
        per_image_ms = round(stats_wall["median"] / b_sz, 3)

        # Forward-only throughput (excluding image preprocessing and postprocessing)
        fwd_throughput_fps = round((b_sz / (stats_forward["median"] / 1000.0)), 1) if stats_forward["median"] > 0 else 0.0

        batch_results[b_sz] = {
            "batch_size": b_sz,
            "throughput_fps": throughput_fps,
            "per_image_ms": per_image_ms,
            "forward_only_throughput_fps": fwd_throughput_fps,
            "wall_clock_stats": stats_wall,
            "engine_total_stats": stats_engine,
            "forward_stats": stats_forward,
            "preprocess_stats": stats_preprocess,
            "postprocess_stats": stats_postprocess,
            "raw_measurements": {
                "wall_latencies_ms": [round(x, 3) for x in wall_latencies_ms],
                "forward_ms": [round(x, 3) for x in forward_ms],
            },
        }

    return {
        "model_name": model_name,
        "precision": precision,
        "device": device,
        "backend": meta["backend"],
        "model_dtype": meta["model_dtype"],
        "is_diagnostic": cfg.get("is_diagnostic", False),
        "note": cfg.get("note", ""),
        "load_time_ms": load_time_ms,
        "artifact_path": str(artifact_path),
        "artifact_size_bytes": meta.get("artifact_size_bytes", 0),
        "artifact_sha256": artifact_sha,
        "parameter_count": meta.get("parameter_count", 0),
        "batch_benchmarks": batch_results,
    }


def parse_batch_sizes(val: str) -> List[int]:
    """Parses a comma-separated string of batch sizes."""
    try:
        sizes = [int(s.strip()) for s in val.split(",") if s.strip()]
        if not sizes or any(s < 1 for s in sizes):
            raise ValueError()
        return sizes
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid batch sizes '{val}'. Expected positive integers (e.g. '1,8,32,64')."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds argument parser for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Edge Vision Reliability Lab — Inference Benchmark Runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18_cifar", "mobilenet_v3_small_cifar"],
        help="List of model architectures to benchmark.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "fp16", "int8"],
        choices=["fp32", "fp16", "int8"],
        help="List of precisions to benchmark.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "mps"],
        choices=["cpu", "mps"],
        help="List of target devices to evaluate where compatible.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default="1,8,32,64",
        help="Comma-separated batch sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of untimed warmup iterations per batch size.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="Number of timed repetitions per batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthetic input generation.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        default=False,
        help="Skip CPU FP16 diagnostic configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save benchmark_summary.csv and benchmark_results.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing benchmark artifacts in destination directory.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Fast sanity check: batch sizes [1, 8], warmup=1, repetitions=2.",
    )
    return parser


def main() -> int:
    """Benchmark runner CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    has_mps = torch.backends.mps.is_available()

    batch_sizes = [1, 8] if args.smoke_test else args.batch_sizes
    warmup = 1 if args.smoke_test else args.warmup
    repetitions = 2 if args.smoke_test else args.repetitions

    # Output directory resolution and collision check
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("experiments/inference_benchmarks") / f"run_{timestamp_str}"
        out_dir = out_dir.resolve()

    csv_path = out_dir / "benchmark_summary.csv"
    json_path = out_dir / "benchmark_results.json"

    if not args.overwrite and (csv_path.exists() or json_path.exists()):
        sys.stderr.write(
            f"Error: Output directory already contains benchmark artifacts: '{out_dir}'.\n"
            "Use --overwrite to allow overwriting or specify a different --output-dir.\n"
        )
        return 1

    # Filter applicable configurations based on args and hardware capabilities
    configs_to_run = []
    for cfg in BENCHMARK_CONFIGURATIONS:
        if cfg["model"] not in args.models:
            continue
        if cfg["precision"] not in args.precisions:
            continue
        if cfg["device"] not in args.devices:
            continue
        if cfg["is_diagnostic"] and args.skip_diagnostics:
            continue
        if cfg["device"] == "mps" and not has_mps:
            sys.stdout.write(f"Skipping MPS configuration ({cfg['model']} {cfg['precision']}): MPS not available.\n")
            continue
        configs_to_run.append(cfg)

    if not configs_to_run:
        sys.stderr.write("Error: No valid benchmark configurations match the specified filters.\n")
        return 1

    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("Edge Vision Reliability Lab — Inference Benchmarking\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write(f"Platform:          {platform.platform()} ({platform.machine()})\n")
    sys.stdout.write(f"PyTorch:           {torch.__version__}\n")
    sys.stdout.write(f"Apple MPS Support: {'Available' if has_mps else 'Unavailable'}\n")
    sys.stdout.write(f"Configurations:    {len(configs_to_run)} variants to benchmark\n")
    sys.stdout.write(f"Batch Sizes:       {batch_sizes}\n")
    sys.stdout.write(f"Warmup / Reps:     {warmup} warmup / {repetitions} repetitions per size\n")
    sys.stdout.write(f"Destination:       {out_dir}\n")
    sys.stdout.write("=" * 70 + "\n\n")

    benchmark_runs: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs_to_run, 1):
        m = cfg["model"]
        p = cfg["precision"]
        d = cfg["device"]
        diag = " [DIAGNOSTIC]" if cfg["is_diagnostic"] else ""
        sys.stdout.write(f"[{idx}/{len(configs_to_run)}] Benchmarking {m} [{p.upper()}] on {d.upper()}{diag}...\n")

        try:
            res = run_benchmark_configuration(
                cfg=cfg,
                batch_sizes=batch_sizes,
                warmup=warmup,
                repetitions=repetitions,
                seed=args.seed,
            )
            benchmark_runs.append(res)

            for b_sz, b_data in res["batch_benchmarks"].items():
                w_stats = b_data["wall_clock_stats"]
                f_stats = b_data["forward_stats"]
                csv_rows.append({
                    "model": m,
                    "precision": p,
                    "device": d,
                    "backend": res["backend"],
                    "batch_size": b_sz,
                    "throughput_fps": b_data["throughput_fps"],
                    "median_total_ms": w_stats["median"],
                    "mean_total_ms": w_stats["mean"],
                    "p95_total_ms": w_stats["p95"],
                    "std_total_ms": w_stats["std"],
                    "per_image_ms": b_data["per_image_ms"],
                    "median_forward_ms": f_stats["median"],
                    "forward_only_throughput_fps": b_data["forward_only_throughput_fps"],
                    "load_time_ms": res["load_time_ms"],
                    "is_diagnostic": res["is_diagnostic"],
                    "note": res["note"],
                })

            # Print concise progress line for batch size 1 and max batch size
            b1 = res["batch_benchmarks"].get(1, {})
            b_max = res["batch_benchmarks"].get(max(batch_sizes), {})
            b1_lat = b1.get("wall_clock_stats", {}).get("median", 0.0)
            bmax_fps = b_max.get("throughput_fps", 0.0)
            sys.stdout.write(f"   -> Single Image: {b1_lat:.2f} ms | Batch {max(batch_sizes)}: {bmax_fps:.1f} fps\n")

        except Exception as err:
            sys.stderr.write(f"Error benchmarking {m} [{p}] on {d}: {err}\n")
            return 1

    # Save output artifacts
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    csv_fieldnames = [
        "model",
        "precision",
        "device",
        "backend",
        "batch_size",
        "throughput_fps",
        "median_total_ms",
        "mean_total_ms",
        "p95_total_ms",
        "std_total_ms",
        "per_image_ms",
        "median_forward_ms",
        "forward_only_throughput_fps",
        "load_time_ms",
        "is_diagnostic",
        "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 2. Save JSON
    full_payload = {
        "benchmark_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit_hash(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "mps_available": has_mps,
            "random_seed": args.seed,
            "warmup_iterations": warmup,
            "measured_repetitions": repetitions,
            "tested_batch_sizes": batch_sizes,
            "measurement_notes": (
                "Steady-state latency excludes model loading time. "
                "Wall-clock timing includes preprocessing, forward pass, and postprocessing. "
                "Forward-only timing isolates model execution with device synchronization. "
                "Apple Silicon CPU benchmarks do not generalize to other CPU microarchitectures or x86."
            ),
        },
        "summary_table": csv_rows,
        "detailed_runs": benchmark_runs,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    sys.stdout.write("\n" + "=" * 70 + "\n")
    sys.stdout.write("Benchmarking Completed Successfully\n")
    sys.stdout.write("=" * 70 + "\n")
    sys.stdout.write(f"Tabular CSV:  {csv_path}\n")
    sys.stdout.write(f"Detailed JSON:{json_path}\n")
    sys.stdout.write("=" * 70 + "\n\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
