"""
FP32 model robustness evaluation engine under clean and corrupted conditions.

Evaluates multiple models against identical batches of clean and synthetically
corrupted images, ensuring exact input identity across architectures and calculating
fine-grained and aggregate robustness metrics.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets

from edge_vision.corruptions import (
    CorruptedDataset,
    get_corruption,
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
from edge_vision.evaluation.metrics import EvaluationMetrics, EvaluationTracker


@dataclass(frozen=True)
class RobustnessCondition:
    """Specification of an evaluation condition (clean or corrupted)."""
    condition_id: str
    is_clean: bool
    corruption_name: Optional[str] = None
    severity: Optional[int] = None
    description: str = ""


@dataclass
class ConditionModelResult:
    """
    Metrics and degradation for a specific model under a specific condition.

    Note on timing:
      condition_wall_time_seconds represents the shared total condition wall-clock time,
      covering DataLoader iteration, CPU-side synthetic corruption synthesis, evaluation
      transforms, host-to-device transfers, and forward passes for both models combined.
      It is not an isolated per-model inference latency measurement.
    """
    model_name: str
    condition_id: str
    is_clean: bool
    corruption_name: Optional[str]
    severity: Optional[int]
    metrics: EvaluationMetrics
    clean_accuracy: float
    accuracy_drop_pp: float
    relative_accuracy_drop: float
    condition_wall_time_seconds: float


@dataclass
class RobustnessAggregateSummary:
    """
    High-level summary of model robustness across clean and corrupted conditions.

    Note on terminology:
      mean_relative_accuracy_drop represents the unweighted arithmetic mean of
      relative accuracy degradation ((clean_acc - corrupt_acc) / clean_acc) across
      corrupted conditions. It is not equivalent to standard mean Corruption Error (mCE),
      which normalizes absolute error against an external baseline architecture.
    """
    clean_accuracy: float
    mean_corrupted_accuracy: float
    mean_accuracy_drop_pp: float
    mean_relative_accuracy_drop: float
    per_corruption_summary: Dict[str, Dict[str, float]]
    per_severity_summary: Dict[int, Dict[str, float]]


@dataclass
class RobustnessBenchmarkResult:
    """Complete container for a multi-model robustness benchmark run."""
    metadata: Dict[str, Any]
    conditions: List[RobustnessCondition]
    results: List[ConditionModelResult]
    summaries: Dict[str, RobustnessAggregateSummary]


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Computes the SHA-256 hexadecimal digest of a file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_standard_robustness_conditions(
    corruption_names: Optional[Sequence[str]] = None,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    include_clean: bool = True,
) -> List[RobustnessCondition]:
    """
    Constructs the list of standard benchmark conditions: clean followed by
    all specified corruptions at all specified severity levels.
    """
    conditions: List[RobustnessCondition] = []

    if include_clean:
        conditions.append(
            RobustnessCondition(
                condition_id="clean",
                is_clean=True,
                corruption_name=None,
                severity=None,
                description="Clean, uncorrupted test partition baseline",
            )
        )

    if corruption_names is None:
        corruption_names = list_corruptions()

    for name in corruption_names:
        defn = get_corruption(name)
        for sev in severities:
            if sev not in defn.severities:
                raise ValueError(f"Severity {sev} not available for corruption '{name}'")
            param_val = defn.severities[sev]
            cond_id = f"{name}_sev{sev}"
            desc = f"{defn.description} ({defn.parameter_name}={param_val})"
            conditions.append(
                RobustnessCondition(
                    condition_id=cond_id,
                    is_clean=False,
                    corruption_name=name,
                    severity=sev,
                    description=desc,
                )
            )

    return conditions


def compute_accuracy_degradation(
    clean_accuracy: float,
    corrupted_accuracy: float,
) -> Tuple[float, float]:
    """
    Computes absolute accuracy drop in percentage points and relative accuracy drop.

    Formula:
      accuracy_drop_pp = (clean_accuracy - corrupted_accuracy) * 100.0
      relative_accuracy_drop = (clean_accuracy - corrupted_accuracy) / clean_accuracy if clean_accuracy > 0 else 0.0

    Note: Negative values are preserved (e.g. if corrupted accuracy slightly exceeds clean).
    """
    drop_pp = (clean_accuracy - corrupted_accuracy) * 100.0
    rel_drop = (clean_accuracy - corrupted_accuracy) / clean_accuracy if clean_accuracy > 0.0 else 0.0
    return drop_pp, rel_drop


def compute_model_robustness_summary(
    results: List[ConditionModelResult],
    model_name: str,
) -> RobustnessAggregateSummary:
    """
    Aggregates condition-level metrics for a model into overall, per-corruption,
    and per-severity summary statistics.
    """
    model_results = [r for r in results if r.model_name == model_name]
    clean_res = next((r for r in model_results if r.is_clean), None)
    clean_acc = clean_res.metrics.accuracy if clean_res is not None else 0.0

    corrupted_results = [r for r in model_results if not r.is_clean]
    if not corrupted_results:
        return RobustnessAggregateSummary(
            clean_accuracy=clean_acc,
            mean_corrupted_accuracy=clean_acc,
            mean_accuracy_drop_pp=0.0,
            mean_relative_accuracy_drop=0.0,
            per_corruption_summary={},
            per_severity_summary={},
        )

    mean_corr_acc = float(np.mean([r.metrics.accuracy for r in corrupted_results]))
    mean_drop_pp = float(np.mean([r.accuracy_drop_pp for r in corrupted_results]))
    mean_rel_drop = float(np.mean([r.relative_accuracy_drop for r in corrupted_results]))

    # Group by corruption
    per_corr: Dict[str, Dict[str, float]] = {}
    corruption_names = sorted(set(r.corruption_name for r in corrupted_results if r.corruption_name))
    for c_name in corruption_names:
        c_res = [r for r in corrupted_results if r.corruption_name == c_name]
        per_corr[c_name] = {
            "mean_accuracy": float(np.mean([r.metrics.accuracy for r in c_res])),
            "mean_accuracy_drop_pp": float(np.mean([r.accuracy_drop_pp for r in c_res])),
            "mean_relative_drop": float(np.mean([r.relative_accuracy_drop for r in c_res])),
        }

    # Group by severity
    per_sev: Dict[int, Dict[str, float]] = {}
    severity_levels = sorted(set(r.severity for r in corrupted_results if r.severity is not None))
    for s_level in severity_levels:
        s_res = [r for r in corrupted_results if r.severity == s_level]
        per_sev[s_level] = {
            "mean_accuracy": float(np.mean([r.metrics.accuracy for r in s_res])),
            "mean_accuracy_drop_pp": float(np.mean([r.accuracy_drop_pp for r in s_res])),
            "mean_relative_drop": float(np.mean([r.relative_accuracy_drop for r in s_res])),
        }

    return RobustnessAggregateSummary(
        clean_accuracy=clean_acc,
        mean_corrupted_accuracy=mean_corr_acc,
        mean_accuracy_drop_pp=mean_drop_pp,
        mean_relative_accuracy_drop=mean_rel_drop,
        per_corruption_summary=per_corr,
        per_severity_summary=per_sev,
    )


class RobustnessEvaluator:
    """
    Orchestrates multi-model robustness evaluation with guaranteed identical input presentation.

    Invariants:
      1. Batch-Level Input Identity: In each condition, each batch of images is generated once
         and evaluated sequentially across all registered models before fetching the next batch.
      2. Zero Memory Duplication: Corrupted images are generated on-the-fly and immediately discarded
         after evaluating both models; the full dataset is never held in memory.
      3. Strict Inference Mode: Models are set to eval() mode and executed under torch.inference_mode().
      4. Synchronized Timing: Accurate wall-clock timing using device synchronization.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        class_names: Sequence[str] = CIFAR10_CLASSES,
    ):
        self.models = models
        self.device = device
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.class_names = class_names

        # Move and freeze models
        for name, model in self.models.items():
            model.eval()
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False

    def evaluate_shared_batch_dataloader(
        self,
        dataloader: DataLoader,
        condition: RobustnessCondition,
        clean_accuracies: Dict[str, float],
        max_samples: Optional[int] = None,
    ) -> Dict[str, ConditionModelResult]:
        """
        Evaluates all models on the exact same batches from dataloader.
        """
        trackers = {m_name: EvaluationTracker(class_names=self.class_names) for m_name in self.models}
        samples_processed = 0

        # Synchronize before timing
        if self.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            for batch_inputs, batch_targets in dataloader:
                batch_size = batch_inputs.size(0)
                if max_samples is not None and samples_processed + batch_size > max_samples:
                    remaining = max_samples - samples_processed
                    batch_inputs = batch_inputs[:remaining]
                    batch_targets = batch_targets[:remaining]
                    batch_size = remaining

                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)

                # Present the identical batch_inputs tensor to each model
                for m_name, model in self.models.items():
                    outputs = model(batch_inputs)
                    loss = self.criterion(outputs, batch_targets)
                    trackers[m_name].update(
                        batch_loss=loss.item(),
                        preds=outputs,
                        targets=batch_targets,
                        batch_size=batch_size,
                    )

                samples_processed += batch_size
                if max_samples is not None and samples_processed >= max_samples:
                    break

        if self.device.type == "mps":
            torch.mps.synchronize()
        total_time = time.perf_counter() - t0

        results: Dict[str, ConditionModelResult] = {}
        for m_name, tracker in trackers.items():
            metrics = tracker.compute()
            clean_acc = clean_accuracies.get(m_name, metrics.accuracy if condition.is_clean else 0.0)
            drop_pp, rel_drop = compute_accuracy_degradation(clean_acc, metrics.accuracy)

            results[m_name] = ConditionModelResult(
                model_name=m_name,
                condition_id=condition.condition_id,
                is_clean=condition.is_clean,
                corruption_name=condition.corruption_name,
                severity=condition.severity,
                metrics=metrics,
                clean_accuracy=clean_acc,
                accuracy_drop_pp=drop_pp,
                relative_accuracy_drop=rel_drop,
                condition_wall_time_seconds=total_time,
            )

        return results


def construct_condition_dataloader(
    base_dataset: Dataset,
    condition: RobustnessCondition,
    transform: Any,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Constructs a memory-conscious DataLoader for a clean or corrupted condition.
    """
    if condition.is_clean:
        # Wrap in a lightweight view applying the evaluation transform directly
        from edge_vision.data.dataset import TransformedSubset
        if hasattr(base_dataset, "data") and hasattr(base_dataset, "targets"):
            indices = list(range(len(base_dataset)))
            dataset = TransformedSubset(base_dataset, indices, transform=transform)
        else:
            dataset = base_dataset
    else:
        # Use existing CorruptedDataset adapter
        dataset = CorruptedDataset(
            base_dataset=base_dataset,
            corruption_name=condition.corruption_name,
            severity=condition.severity,
            transform=transform,
            seed=seed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


REQUIRED_BENCHMARK_ARTIFACTS = (
    "metadata.json",
    "results.csv",
    "results.json",
    "summary.json",
)


def check_existing_benchmark(output_dir: Union[str, Path]) -> bool:
    """
    Checks whether an output directory already contains a completed benchmark.
    Returns True if any of the core benchmark artifacts are present.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        return False
    return any((out_path / artifact).is_file() for artifact in REQUIRED_BENCHMARK_ARTIFACTS)


def validate_output_directory(output_dir: Union[str, Path], overwrite: bool = False) -> None:
    """
    Validates whether the target output directory is safe for benchmark execution.

    Case 1: Output directory does not exist -> Valid.
    Case 2: Output directory exists but contains no benchmark artifacts -> Valid.
    Case 3: Output directory contains completed benchmark artifacts and overwrite=False -> Raises FileExistsError.
    Case 4: Output directory contains completed benchmark artifacts and overwrite=True -> Valid.

    Raises:
        FileExistsError: If output_dir already contains benchmark artifacts and overwrite is False.
    """
    out_path = Path(output_dir)
    if check_existing_benchmark(out_path) and not overwrite:
        existing = [a for a in REQUIRED_BENCHMARK_ARTIFACTS if (out_path / a).is_file()]
        raise FileExistsError(
            f"Output directory '{out_path}' already contains completed benchmark artifacts ({', '.join(existing)}). "
            f"Pass '--overwrite' to explicitly permit replacing existing results."
        )


def save_robustness_results(
    benchmark_result: RobustnessBenchmarkResult,
    output_dir: Union[str, Path],
) -> Path:
    """
    Persists structured benchmark outputs atomically:
      - metadata.json
      - results.csv (one row per model-condition)
      - results.json (complete fine-grained metrics, confusion matrices)
      - summary.json (aggregate metrics and comparisons)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. metadata.json
    meta_file = out_path / "metadata.json"
    temp_meta = meta_file.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(benchmark_result.metadata, f, indent=2, sort_keys=True)
    os.replace(temp_meta, meta_file)

    # 2. results.csv
    csv_rows = []
    for r in benchmark_result.results:
        csv_rows.append({
            "model": r.model_name,
            "condition": r.condition_id,
            "corruption": r.corruption_name if r.corruption_name is not None else "none",
            "severity": r.severity if r.severity is not None else 0,
            "sample_count": r.metrics.total_samples,
            "loss": round(r.metrics.loss, 6),
            "accuracy": round(r.metrics.accuracy, 6),
            "correct": r.metrics.correct_predictions,
            "incorrect": r.metrics.total_samples - r.metrics.correct_predictions,
            "clean_accuracy": round(r.clean_accuracy, 6),
            "accuracy_drop_pp": round(r.accuracy_drop_pp, 4),
            "relative_accuracy_drop": round(r.relative_accuracy_drop, 6),
            "condition_wall_time_seconds": round(r.condition_wall_time_seconds, 3),
        })

    df = pd.DataFrame(csv_rows)
    csv_file = out_path / "results.csv"
    temp_csv = csv_file.with_suffix(f".tmp.{os.getpid()}")
    df.to_csv(temp_csv, index=False)
    os.replace(temp_csv, csv_file)

    # 3. results.json
    results_dict = {
        "metadata": benchmark_result.metadata,
        "conditions": [asdict(c) for c in benchmark_result.conditions],
        "evaluations": [],
    }
    for r in benchmark_result.results:
        results_dict["evaluations"].append({
            "model_name": r.model_name,
            "condition_id": r.condition_id,
            "is_clean": r.is_clean,
            "corruption_name": r.corruption_name,
            "severity": r.severity,
            "clean_accuracy": r.clean_accuracy,
            "accuracy_drop_pp": r.accuracy_drop_pp,
            "relative_accuracy_drop": r.relative_accuracy_drop,
            "condition_wall_time_seconds": r.condition_wall_time_seconds,
            "metrics": asdict(r.metrics),
        })

    json_file = out_path / "results.json"
    temp_json = json_file.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    os.replace(temp_json, json_file)

    # 4. summary.json
    summary_dict = {
        "metadata": {
            "timestamp": benchmark_result.metadata.get("timestamp_end"),
            "models": list(benchmark_result.summaries.keys()),
            "total_conditions": len(benchmark_result.conditions),
        },
        "model_summaries": {m: asdict(s) for m, s in benchmark_result.summaries.items()},
    }
    # Add architecture comparison delta if both models present
    models = list(benchmark_result.summaries.keys())
    if len(models) == 2:
        m1, m2 = models[0], models[1]
        s1, s2 = benchmark_result.summaries[m1], benchmark_result.summaries[m2]
        summary_dict["comparative_delta"] = {
            f"{m1}_vs_{m2}_clean_acc_diff": s1.clean_accuracy - s2.clean_accuracy,
            f"{m1}_vs_{m2}_mean_corrupt_acc_diff": s1.mean_corrupted_accuracy - s2.mean_corrupted_accuracy,
            f"{m1}_vs_{m2}_mean_drop_diff_pp": s1.mean_accuracy_drop_pp - s2.mean_accuracy_drop_pp,
        }

    summary_file = out_path / "summary.json"
    temp_summary = summary_file.with_suffix(f".tmp.{os.getpid()}")
    with open(temp_summary, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    os.replace(temp_summary, summary_file)

    return out_path
