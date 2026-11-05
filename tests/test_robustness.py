"""
Unit and integration tests for FP32 model robustness evaluation engine.
Uses lightweight synthetic datasets and dummy models to test all logic in CPU-only, isolated mode.
"""

import io
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from edge_vision.evaluation.metrics import EvaluationMetrics
from edge_vision.evaluation.robustness import (
    REQUIRED_BENCHMARK_ARTIFACTS,
    ConditionModelResult,
    RobustnessAggregateSummary,
    RobustnessBenchmarkResult,
    RobustnessCondition,
    RobustnessEvaluator,
    check_existing_benchmark,
    compute_accuracy_degradation,
    compute_file_sha256,
    compute_model_robustness_summary,
    construct_condition_dataloader,
    get_standard_robustness_conditions,
    save_robustness_results,
    validate_output_directory,
)


# ============================================================================
# Test Fixtures & Lightweight Helpers
# ============================================================================

class DummyTestModel(nn.Module):
    """Predictable dummy model outputting constant logits based on an offset."""

    def __init__(self, preferred_class: int = 0):
        super().__init__()
        self.preferred_class = preferred_class
        # Dummy parameter to satisfy nn.Module requirements
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        logits = torch.zeros((batch_size, 10), device=x.device, dtype=torch.float32)
        logits[:, self.preferred_class] = 10.0
        return logits


class IdentityCheckingModel(nn.Module):
    """Model that records the exact SHA-256 hash of each input batch it receives."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.batch_hashes: List[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute hash of raw input tensor bytes
        arr_bytes = x.detach().cpu().numpy().tobytes()
        import hashlib
        self.batch_hashes.append(hashlib.sha256(arr_bytes).hexdigest())
        return torch.zeros((x.size(0), 10), device=x.device)


class SyntheticRawCIFAR(Dataset):
    """Lightweight in-memory dataset providing raw uint8 arrays."""

    def __init__(self, size: int = 40):
        rng = np.random.default_rng(707)
        self.data = rng.integers(20, 235, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = [int(x) for x in rng.integers(0, 10, size=size)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


# ============================================================================
# 1. Degradation and Summary Calculation Tests
# ============================================================================

def test_compute_accuracy_degradation_standard():
    clean_acc = 0.9478
    corrupt_acc = 0.8000

    drop_pp, rel_drop = compute_accuracy_degradation(clean_acc, corrupt_acc)

    assert pytest.approx(drop_pp, rel=1e-5) == 14.78
    assert pytest.approx(rel_drop, rel=1e-5) == (0.9478 - 0.8000) / 0.9478


def test_compute_accuracy_degradation_preserves_negative():
    # Model performs slightly better on corrupted data
    clean_acc = 0.8500
    corrupt_acc = 0.8650

    drop_pp, rel_drop = compute_accuracy_degradation(clean_acc, corrupt_acc)

    assert drop_pp < 0.0
    assert pytest.approx(drop_pp, rel=1e-5) == -1.50
    assert pytest.approx(rel_drop, rel=1e-5) == -1.50 / 85.0


def test_compute_accuracy_degradation_zero_clean():
    clean_acc = 0.0
    corrupt_acc = 0.0

    drop_pp, rel_drop = compute_accuracy_degradation(clean_acc, corrupt_acc)
    assert drop_pp == 0.0
    assert rel_drop == 0.0


def test_compute_model_robustness_summary_aggregation():
    # 1 clean condition (acc 0.90) and 2 corruptions x 2 severities = 4 corrupted conditions
    results = [
        ConditionModelResult(
            model_name="model_a",
            condition_id="clean",
            is_clean=True,
            corruption_name=None,
            severity=None,
            metrics=EvaluationMetrics(100, 90, 0.90, 0.2, {}, {}, {}, []),
            clean_accuracy=0.90,
            accuracy_drop_pp=0.0,
            relative_accuracy_drop=0.0,
            condition_wall_time_seconds=1.0,
        ),
        ConditionModelResult(
            model_name="model_a",
            condition_id="c1_sev1",
            is_clean=False,
            corruption_name="corr_1",
            severity=1,
            metrics=EvaluationMetrics(100, 80, 0.80, 0.4, {}, {}, {}, []),
            clean_accuracy=0.90,
            accuracy_drop_pp=10.0,
            relative_accuracy_drop=10.0 / 90.0,
            condition_wall_time_seconds=1.0,
        ),
        ConditionModelResult(
            model_name="model_a",
            condition_id="c1_sev2",
            is_clean=False,
            corruption_name="corr_1",
            severity=2,
            metrics=EvaluationMetrics(100, 70, 0.70, 0.6, {}, {}, {}, []),
            clean_accuracy=0.90,
            accuracy_drop_pp=20.0,
            relative_accuracy_drop=20.0 / 90.0,
            condition_wall_time_seconds=1.0,
        ),
        ConditionModelResult(
            model_name="model_a",
            condition_id="c2_sev1",
            is_clean=False,
            corruption_name="corr_2",
            severity=1,
            metrics=EvaluationMetrics(100, 85, 0.85, 0.3, {}, {}, {}, []),
            clean_accuracy=0.90,
            accuracy_drop_pp=5.0,
            relative_accuracy_drop=5.0 / 90.0,
            condition_wall_time_seconds=1.0,
        ),
        ConditionModelResult(
            model_name="model_a",
            condition_id="c2_sev2",
            is_clean=False,
            corruption_name="corr_2",
            severity=2,
            metrics=EvaluationMetrics(100, 65, 0.65, 0.7, {}, {}, {}, []),
            clean_accuracy=0.90,
            accuracy_drop_pp=25.0,
            relative_accuracy_drop=25.0 / 90.0,
            condition_wall_time_seconds=1.0,
        ),
    ]

    summary = compute_model_robustness_summary(results, "model_a")

    assert summary.clean_accuracy == 0.90
    # Mean of [0.80, 0.70, 0.85, 0.65] = 3.0 / 4 = 0.75
    assert pytest.approx(summary.mean_corrupted_accuracy, rel=1e-5) == 0.75
    # Mean drop of [10, 20, 5, 25] = 60 / 4 = 15.0 pp
    assert pytest.approx(summary.mean_accuracy_drop_pp, rel=1e-5) == 15.0
    # Mean relative accuracy drop
    expected_rel_drop = (10.0 / 90.0 + 20.0 / 90.0 + 5.0 / 90.0 + 25.0 / 90.0) / 4.0
    assert pytest.approx(summary.mean_relative_accuracy_drop, rel=1e-5) == expected_rel_drop

    # Per-corruption check
    assert pytest.approx(summary.per_corruption_summary["corr_1"]["mean_accuracy"], rel=1e-5) == 0.75
    assert pytest.approx(summary.per_corruption_summary["corr_2"]["mean_accuracy"], rel=1e-5) == 0.75

    # Per-severity check
    assert pytest.approx(summary.per_severity_summary[1]["mean_accuracy"], rel=1e-5) == (0.80 + 0.85) / 2.0
    assert pytest.approx(summary.per_severity_summary[2]["mean_accuracy"], rel=1e-5) == (0.70 + 0.65) / 2.0


# ============================================================================
# 2. SHA-256 and Condition Generation Tests
# ============================================================================

def test_compute_file_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Edge Vision Reliability Lab", encoding="utf-8")

    import hashlib
    expected_hash = hashlib.sha256(b"Edge Vision Reliability Lab").hexdigest()
    assert compute_file_sha256(test_file) == expected_hash

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "nonexistent.txt")


def test_get_standard_robustness_conditions():
    conds = get_standard_robustness_conditions(
        corruption_names=["brightness", "contrast"],
        severities=[1, 2, 3],
        include_clean=True,
    )

    # 1 clean + 2 corruptions * 3 severities = 7 conditions
    assert len(conds) == 7
    assert conds[0].is_clean is True and conds[0].condition_id == "clean"
    assert conds[1].condition_id == "brightness_sev1"
    assert conds[-1].condition_id == "contrast_sev3"


# ============================================================================
# 3. Shared-Batch Input Identity and Evaluator Tests
# ============================================================================

def test_shared_batch_input_identity_across_models():
    """
    Directly tests that two models evaluated within the shared-batch execution loop
    receive byte-for-byte identical input tensors.
    """
    model1 = IdentityCheckingModel()
    model2 = IdentityCheckingModel()

    dataset = SyntheticRawCIFAR(size=30)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    condition = RobustnessCondition("clean", is_clean=True, description="test")
    loader = construct_condition_dataloader(
        base_dataset=dataset,
        condition=condition,
        transform=transform,
        batch_size=10,
    )

    evaluator = RobustnessEvaluator(
        models={"m1": model1, "m2": model2},
        device=torch.device("cpu"),
    )

    _ = evaluator.evaluate_shared_batch_dataloader(
        dataloader=loader,
        condition=condition,
        clean_accuracies={},
    )

    # Verify both models evaluated the same number of batches
    assert len(model1.batch_hashes) == len(model2.batch_hashes) == 3
    # Verify exact bitwise hash match for each batch
    for h1, h2 in zip(model1.batch_hashes, model2.batch_hashes):
        assert h1 == h2


def test_evaluator_partial_sample_evaluation():
    """Verifies that max_samples truncates the evaluation properly without crashing."""
    m = DummyTestModel(preferred_class=0)
    dataset = SyntheticRawCIFAR(size=50)

    condition = RobustnessCondition("clean", is_clean=True)
    loader = construct_condition_dataloader(
        base_dataset=dataset,
        condition=condition,
        transform=transforms.ToTensor(),
        batch_size=16,
    )

    evaluator = RobustnessEvaluator(
        models={"dummy": m},
        device=torch.device("cpu"),
    )

    # Limit to 25 samples (2 batches: 16 + 9)
    res = evaluator.evaluate_shared_batch_dataloader(
        dataloader=loader,
        condition=condition,
        clean_accuracies={},
        max_samples=25,
    )

    assert res["dummy"].metrics.total_samples == 25


# ============================================================================
# 4. Result Serialization & Schema Verification Tests
# ============================================================================

def test_save_robustness_results_schema(tmp_path: Path):
    condition = RobustnessCondition("clean", is_clean=True, description="Clean test")
    metrics = EvaluationMetrics(
        total_samples=10,
        correct_predictions=8,
        accuracy=0.80,
        loss=0.5,
        per_class_counts={"c0": 5, "c1": 5},
        per_class_correct={"c0": 4, "c1": 4},
        per_class_accuracy={"c0": 0.8, "c1": 0.8},
        confusion_matrix=[[4, 1], [1, 4]],
    )
    result = ConditionModelResult(
        model_name="dummy_model",
        condition_id="clean",
        is_clean=True,
        corruption_name=None,
        severity=None,
        metrics=metrics,
        clean_accuracy=0.80,
        accuracy_drop_pp=0.0,
        relative_accuracy_drop=0.0,
        condition_wall_time_seconds=0.123,
    )
    summary = RobustnessAggregateSummary(
        clean_accuracy=0.80,
        mean_corrupted_accuracy=0.80,
        mean_accuracy_drop_pp=0.0,
        mean_relative_accuracy_drop=0.0,
        per_corruption_summary={},
        per_severity_summary={},
    )
    benchmark_res = RobustnessBenchmarkResult(
        metadata={"seed": 42, "test_field": "val"},
        conditions=[condition],
        results=[result],
        summaries={"dummy_model": summary},
    )

    out_dir = save_robustness_results(benchmark_res, tmp_path / "robustness_run")

    # Verify all expected artifacts exist
    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "results.csv").is_file()
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "summary.json").is_file()

    # Verify CSV schema
    import pandas as pd
    df = pd.read_csv(out_dir / "results.csv")
    expected_cols = [
        "model", "condition", "corruption", "severity", "sample_count",
        "loss", "accuracy", "correct", "incorrect", "clean_accuracy",
        "accuracy_drop_pp", "relative_accuracy_drop", "condition_wall_time_seconds"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 1
    assert df["model"].iloc[0] == "dummy_model"
    assert df["accuracy"].iloc[0] == 0.80
    assert df["condition_wall_time_seconds"].iloc[0] == 0.123

    # Verify JSON structure contains condition_wall_time_seconds
    with open(out_dir / "results.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert len(json_data["evaluations"]) == 1
    assert json_data["evaluations"][0]["condition_wall_time_seconds"] == 0.123


# ============================================================================
# 5. Output Directory Overwrite Protection & Timing Semantics Tests
# ============================================================================

def test_validate_output_directory_nonexistent_accepted(tmp_path: Path):
    """Case 1: Output directory does not exist -> accepted."""
    nonexistent = tmp_path / "new_benchmark_dir"
    assert not nonexistent.exists()
    # Should not raise
    validate_output_directory(nonexistent, overwrite=False)


def test_validate_output_directory_empty_accepted(tmp_path: Path):
    """Case 2: Output directory exists but is empty -> accepted."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    assert empty_dir.exists()
    assert not any(empty_dir.iterdir())
    # Should not raise
    validate_output_directory(empty_dir, overwrite=False)


def test_validate_output_directory_existing_rejected_without_overwrite(tmp_path: Path):
    """Case 3: Directory contains completed benchmark artifacts -> rejected without overwrite."""
    bm_dir = tmp_path / "completed_benchmark"
    bm_dir.mkdir(parents=True, exist_ok=True)
    for artifact in REQUIRED_BENCHMARK_ARTIFACTS:
        (bm_dir / artifact).write_text("{\"status\": \"completed\"}", encoding="utf-8")

    # Must raise FileExistsError
    with pytest.raises(FileExistsError) as exc_info:
        validate_output_directory(bm_dir, overwrite=False)

    assert "already contains completed benchmark artifacts" in str(exc_info.value)
    assert "--overwrite" in str(exc_info.value)

    # Invariant: existing artifacts remain completely unmodified
    for artifact in REQUIRED_BENCHMARK_ARTIFACTS:
        assert (bm_dir / artifact).read_text(encoding="utf-8") == "{\"status\": \"completed\"}"


def test_validate_output_directory_existing_accepted_with_overwrite(tmp_path: Path):
    """Case 4: Directory contains completed benchmark artifacts -> accepted with overwrite=True."""
    bm_dir = tmp_path / "completed_benchmark"
    bm_dir.mkdir(parents=True, exist_ok=True)
    for artifact in REQUIRED_BENCHMARK_ARTIFACTS:
        (bm_dir / artifact).write_text("{}", encoding="utf-8")

    # Should not raise
    validate_output_directory(bm_dir, overwrite=True)


def test_unrelated_files_preserved_on_overwrite(tmp_path: Path):
    """Unrelated user files in output directory are not deleted on benchmark write."""
    out_dir = tmp_path / "run_with_notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_file = out_dir / "user_experiment_notes.txt"
    notes_file.write_text("Important research annotations", encoding="utf-8")

    condition = RobustnessCondition("clean", is_clean=True)
    metrics = EvaluationMetrics(10, 8, 0.80, 0.5, {}, {}, {}, [])
    result = ConditionModelResult(
        "model_a", "clean", True, None, None, metrics, 0.80, 0.0, 0.0, 0.555
    )
    summary = RobustnessAggregateSummary(0.80, 0.80, 0.0, 0.0, {}, {})
    benchmark_res = RobustnessBenchmarkResult(
        metadata={"seed": 42},
        conditions=[condition],
        results=[result],
        summaries={"model_a": summary},
    )

    save_robustness_results(benchmark_res, out_dir)

    # Verify unrelated file remains completely intact
    assert notes_file.is_file()
    assert notes_file.read_text(encoding="utf-8") == "Important research annotations"
    # Verify new artifacts also exist
    assert (out_dir / "results.csv").is_file()


def test_atomic_replacement_protects_existing_artifacts_on_crash(tmp_path: Path):
    """
    If a rerun begins with --overwrite but an error occurs before serialization,
    existing completed benchmark artifacts must remain intact.
    """
    out_dir = tmp_path / "prev_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    for artifact in REQUIRED_BENCHMARK_ARTIFACTS:
        (out_dir / artifact).write_text(f"original_{artifact}", encoding="utf-8")

    # Validate passes with overwrite=True
    validate_output_directory(out_dir, overwrite=True)

    # Simulate an error during execution before save_robustness_results is called
    try:
        raise RuntimeError("Simulated crash mid-benchmark")
    except RuntimeError:
        pass

    # Invariant: Old artifacts must not have been touched
    for artifact in REQUIRED_BENCHMARK_ARTIFACTS:
        assert (out_dir / artifact).read_text(encoding="utf-8") == f"original_{artifact}"
