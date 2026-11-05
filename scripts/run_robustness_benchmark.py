#!/usr/bin/env python3
"""
CLI entry point for FP32 model robustness benchmarking under clean and synthetic image corruptions.
Evaluates ResNet-18 and MobileNetV3-Small baselines on identical corrupted image batches.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import PIL
import scipy
import torch
import torchvision
import torchvision.datasets as datasets

from edge_vision.corruptions import (
    get_default_corruption_config_path,
    list_corruptions,
    load_corruption_config,
)
from edge_vision.data.config import load_data_config
from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.data.transforms import get_cifar10_transforms
from edge_vision.evaluation.evaluator import (
    load_checkpoint_for_evaluation,
    resolve_evaluation_device,
)
from edge_vision.evaluation.robustness import (
    ConditionModelResult,
    RobustnessAggregateSummary,
    RobustnessBenchmarkResult,
    RobustnessCondition,
    RobustnessEvaluator,
    check_existing_benchmark,
    compute_file_sha256,
    compute_model_robustness_summary,
    construct_condition_dataloader,
    get_standard_robustness_conditions,
    save_robustness_results,
    validate_output_directory,
)


DEFAULT_CHECKPOINTS = {
    "resnet18_cifar": "experiments/exp_baseline_resnet18_seed42/checkpoints/best_checkpoint.pt",
    "mobilenet_v3_small_cifar": "experiments/exp_baseline_mobilenet_v3_small_seed42/checkpoints/best_checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FP32 model robustness evaluation on clean and synthetically corrupted CIFAR-10 test imagery."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18_cifar", "mobilenet_v3_small_cifar"],
        help="Model architecture names to benchmark (default: resnet18_cifar mobilenet_v3_small_cifar).",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="List of corruption names to evaluate (defaults to all configured in corruptions.yaml).",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Severity levels to evaluate (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experiment seed for deterministic noise generation (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation dataloaders (default: 64).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional upper bound on test samples evaluated per condition (e.g. 128 for testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/robustness_fp32_seed42",
        help="Directory to store benchmark artifacts (default: experiments/robustness_fp32_seed42).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute a fast smoke test (128 samples, clean + gaussian_noise sev 1).",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Explicit confirmation required to execute the full 620,000-prediction benchmark.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacing existing completed benchmark artifacts in the output directory.",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> Path:
    # 1. Output Directory Overwrite Validation (before any heavy initialization or model loading)
    out_path = Path(args.output_dir)
    try:
        validate_output_directory(out_path, overwrite=args.overwrite)
    except FileExistsError as err:
        print(f"\n[!] OVERWRITE ERROR: {err}", file=sys.stderr)
        print("    Aborting execution to protect existing completed benchmark artifacts.", file=sys.stderr)
        print("    Pass '--overwrite' to explicitly allow replacement.", file=sys.stderr)
        sys.exit(1)

    # 2. Scope / Confirmation Guard
    if args.smoke_test:
        corruptions = ["gaussian_noise"]
        severities = [1]
        max_samples = 128 if args.max_samples is None else args.max_samples
        print(f"[*] Running FAST SMOKE TEST (128 samples, clean + gaussian_noise sev 1)...")
    elif args.max_samples is not None:
        corruptions = args.corruptions if args.corruptions is not None else list_corruptions()
        severities = args.severities
        max_samples = args.max_samples
        print(f"[*] Running BOUNDED BENCHMARK (max {max_samples} samples per condition)...")
    elif args.full_run:
        corruptions = args.corruptions if args.corruptions is not None else list_corruptions()
        severities = args.severities
        max_samples = None
        print(f"[*] Running FULL ROBUSTNESS BENCHMARK (10,000 test samples per condition)...")
    else:
        print("[!] SAFETY STOP: Full robustness benchmark evaluates 620,000 predictions across 62 model-conditions.")
        print("    To launch the full benchmark, pass the explicit '--full-run' flag.")
        print("    To run a fast smoke test, pass '--smoke-test'.")
        print("    To run on a subset of samples, pass '--max-samples <N>'.")
        sys.exit(1)

    t_start = datetime.now(timezone.utc)
    t0_wall = time.perf_counter()

    # 3. Resolve Device & Configs
    device = resolve_evaluation_device(args.device)
    data_cfg = load_data_config("configs/cifar10.yaml")
    corr_cfg_path = get_default_corruption_config_path()
    corr_cfg = load_corruption_config(corr_cfg_path)
    corr_cfg_sha256 = compute_file_sha256(corr_cfg_path)

    print(f"[*] Device: {device} | Experiment Seed: {args.seed} | Batch Size: {args.batch_size}")
    print(f"[*] Corruptions: {corruptions} | Severities: {severities}")

    # 4. Verify and Load Checkpoints
    loaded_models: Dict[str, torch.nn.Module] = {}
    checkpoint_metadata: Dict[str, Any] = {}

    for m_arch in args.models:
        if m_arch not in DEFAULT_CHECKPOINTS:
            raise ValueError(f"Unknown model architecture '{m_arch}'. Expected one of {list(DEFAULT_CHECKPOINTS.keys())}")
        ckpt_path = Path(DEFAULT_CHECKPOINTS[m_arch])
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")

        ckpt_sha256 = compute_file_sha256(ckpt_path)
        model, raw_ckpt = load_checkpoint_for_evaluation(ckpt_path, device=device)
        loaded_models[m_arch] = model

        checkpoint_metadata[m_arch] = {
            "checkpoint_path": str(ckpt_path),
            "sha256": ckpt_sha256,
            "epoch": raw_ckpt.get("epoch"),
            "best_metric_value": raw_ckpt.get("best_metric_value"),
            "best_epoch": raw_ckpt.get("best_epoch"),
            "model_architecture": raw_ckpt.get("model_architecture"),
        }
        print(f"    Loaded {m_arch} (Epoch {raw_ckpt.get('epoch')}, SHA-256: {ckpt_sha256[:12]}...)")

    # 5. Acquire Raw CIFAR-10 Test Dataset
    eval_transform = get_cifar10_transforms(
        train=False,
        augment=False,
        mean=data_cfg.normalization.mean,
        std=data_cfg.normalization.std,
    )
    raw_test_dataset = datasets.CIFAR10(
        root=data_cfg.dataset.raw_dir,
        train=False,
        download=data_cfg.dataset.download,
        transform=None,
    )
    total_test_samples = len(raw_test_dataset)
    print(f"[*] Loaded raw CIFAR-10 test set ({total_test_samples} total images).")

    # 6. Define Conditions
    conditions = get_standard_robustness_conditions(
        corruption_names=corruptions,
        severities=severities,
        include_clean=True,
    )
    print(f"[*] Total conditions to evaluate: {len(conditions)} ({len(conditions) * len(loaded_models)} model-evaluations).")

    # 7. Initialize Evaluator
    evaluator = RobustnessEvaluator(
        models=loaded_models,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        class_names=CIFAR10_CLASSES,
    )

    all_results: List[ConditionModelResult] = []
    clean_accuracies: Dict[str, float] = {}

    # 8. Condition Execution Loop (Shared-Batch Architecture)
    for idx, cond in enumerate(conditions, start=1):
        cond_t0 = time.perf_counter()
        print(f"[{idx:2d}/{len(conditions):2d}] Condition: {cond.condition_id:<22} | {cond.description}")

        loader = construct_condition_dataloader(
            base_dataset=raw_test_dataset,
            condition=cond,
            transform=eval_transform,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=0,
        )

        cond_results = evaluator.evaluate_shared_batch_dataloader(
            dataloader=loader,
            condition=cond,
            clean_accuracies=clean_accuracies,
            max_samples=max_samples,
        )

        # Print progress and record results
        for m_arch, res in cond_results.items():
            all_results.append(res)
            if cond.is_clean:
                clean_accuracies[m_arch] = res.metrics.accuracy
                print(f"      -> {m_arch:<24} Acc: {res.metrics.accuracy * 100:.2f}% | Loss: {res.metrics.loss:.4f}")
            else:
                print(
                    f"      -> {m_arch:<24} Acc: {res.metrics.accuracy * 100:.2f}% "
                    f"(-{res.accuracy_drop_pp:.2f} pp, rel {res.relative_accuracy_drop * 100:.2f}%) | Loss: {res.metrics.loss:.4f}"
                )

    total_wall_time = time.perf_counter() - t0_wall
    t_end = datetime.now(timezone.utc)

    # 9. Compute Aggregate Summaries
    model_summaries: Dict[str, RobustnessAggregateSummary] = {}
    for m_arch in args.models:
        model_summaries[m_arch] = compute_model_robustness_summary(all_results, m_arch)

    # 10. Build Metadata Payload
    metadata = {
        "benchmark_type": "smoke_test" if args.smoke_test else ("bounded" if max_samples else "full"),
        "timestamp_start": t_start.isoformat(),
        "timestamp_end": t_end.isoformat(),
        "total_runtime_seconds": round(total_wall_time, 3),
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_samples_per_condition": max_samples if max_samples else total_test_samples,
        "total_conditions": len(conditions),
        "total_evaluations": len(all_results),
        "dataset": {
            "name": "cifar10",
            "split": "test",
            "total_available_samples": total_test_samples,
            "normalization_mean": data_cfg.normalization.mean,
            "normalization_std": data_cfg.normalization.std,
        },
        "corruption_configuration": {
            "path": str(corr_cfg_path),
            "sha256": corr_cfg_sha256,
            "corruptions_evaluated": corruptions,
            "severities_evaluated": severities,
            "gaussian_noise_crn": True,
        },
        "models_evaluated": checkpoint_metadata,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": getattr(torchvision, "__version__", "unknown"),
            "numpy_version": getattr(np, "__version__", "unknown"),
            "scipy_version": getattr(scipy, "__version__", "unknown"),
            "pillow_version": getattr(PIL, "__version__", "unknown"),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }

    benchmark_result = RobustnessBenchmarkResult(
        metadata=metadata,
        conditions=conditions,
        results=all_results,
        summaries=model_summaries,
    )

    # 11. Persist Outputs Atomically
    out_dir = save_robustness_results(benchmark_result, args.output_dir)
    print(f"\n[+] Benchmark outputs successfully persisted to: {out_dir}")
    print(f"    - {out_dir}/metadata.json")
    print(f"    - {out_dir}/results.csv ({len(all_results)} rows)")
    print(f"    - {out_dir}/results.json")
    print(f"    - {out_dir}/summary.json")

    # 12. Print Executive Summary Table
    print("\n" + "=" * 78)
    print("ROBUSTNESS BENCHMARK SUMMARY")
    print("=" * 78)
    for m_arch, summary in model_summaries.items():
        print(f"Model: {m_arch}")
        print(f"  Clean Accuracy:          {summary.clean_accuracy * 100:.2f}%")
        print(f"  Mean Corrupted Accuracy: {summary.mean_corrupted_accuracy * 100:.2f}%")
        print(f"  Mean Accuracy Drop:      {summary.mean_accuracy_drop_pp:.2f} percentage points")
        print(f"  Mean Relative Drop:      {summary.mean_relative_accuracy_drop * 100:.2f}%")
    print("=" * 78 + "\n")

    return out_dir


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
