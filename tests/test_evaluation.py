"""
Unit and integration tests for standalone checkpoint evaluation infrastructure.
"""

import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_vision.data.dataset import CIFAR10_CLASSES
from edge_vision.evaluation.evaluator import (
    evaluate_model,
    load_checkpoint_for_evaluation,
    resolve_evaluation_device,
    save_evaluation_report,
)
from edge_vision.evaluation.metrics import EvaluationMetrics, EvaluationTracker
from edge_vision.models import get_model, mobilenet_v3_small_cifar, resnet18_cifar
from edge_vision.training.checkpoint import CheckpointManager
from edge_vision.training.trainer import Trainer


# ============================================================================
# 1. Metric Calculation & Confusion Matrix Tests
# ============================================================================

def test_evaluation_tracker_sample_weighted_metrics() -> None:
    tracker = EvaluationTracker(class_names=["class_0", "class_1", "class_2"])

    # Batch 1: 3 samples, targets [0, 1, 2], preds [0, 1, 0] (2 correct, 1 wrong), loss 0.5
    preds1 = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    targets1 = torch.tensor([0, 1, 2])
    tracker.update(batch_loss=0.5, preds=preds1, targets=targets1, batch_size=3)

    # Batch 2: 1 sample (unequal final batch), target [2], pred [2] (1 correct), loss 0.1
    preds2 = torch.tensor([[0.0, 0.0, 5.0]])
    targets2 = torch.tensor([2])
    tracker.update(batch_loss=0.1, preds=preds2, targets=targets2, batch_size=1)

    metrics = tracker.compute()

    # Total samples = 4
    assert metrics.total_samples == 4
    # Correct = 3 (acc = 0.75)
    assert metrics.correct_predictions == 3
    assert pytest.approx(metrics.accuracy, rel=1e-5) == 0.75

    # Sample-weighted loss: (3 * 0.5 + 1 * 0.1) / 4 = 1.6 / 4 = 0.4
    assert pytest.approx(metrics.loss, rel=1e-5) == 0.4

    # Per-class counts and accuracies
    assert metrics.per_class_counts["class_0"] == 1  # 1 sample of class 0
    assert metrics.per_class_correct["class_0"] == 1
    assert pytest.approx(metrics.per_class_accuracy["class_0"], rel=1e-5) == 1.0

    assert metrics.per_class_counts["class_1"] == 1  # 1 sample of class 1
    assert metrics.per_class_correct["class_1"] == 1
    assert pytest.approx(metrics.per_class_accuracy["class_1"], rel=1e-5) == 1.0

    assert metrics.per_class_counts["class_2"] == 2  # 2 samples of class 2
    assert metrics.per_class_correct["class_2"] == 1  # 1 correct, 1 predicted as class 0
    assert pytest.approx(metrics.per_class_accuracy["class_2"], rel=1e-5) == 0.5

    # Confusion matrix checks: 3x3
    cm = metrics.confusion_matrix
    assert len(cm) == 3 and len(cm[0]) == 3
    assert sum(sum(row) for row in cm) == 4
    assert cm[0][0] == 1
    assert cm[1][1] == 1
    assert cm[2][0] == 1  # true 2 predicted as 0
    assert cm[2][2] == 1  # true 2 predicted as 2


def test_empty_evaluation_tracker() -> None:
    tracker = EvaluationTracker(class_names=["c0", "c1"])
    metrics = tracker.compute()
    assert metrics.total_samples == 0
    assert metrics.correct_predictions == 0
    assert metrics.accuracy == 0.0
    assert metrics.loss == 0.0
    assert metrics.per_class_counts["c0"] == 0
    assert metrics.per_class_accuracy["c0"] == 0.0


# ============================================================================
# 2. Checkpoint Loading & Architecture Reconstruction Tests
# ============================================================================

@pytest.mark.parametrize("arch_name,model_fn", [
    ("resnet18_cifar", resnet18_cifar),
    ("mobilenet_v3_small_cifar", mobilenet_v3_small_cifar),
])
def test_checkpoint_reconstruction_and_strict_loading(arch_name: str, model_fn) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = model_fn(num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(tmp_dir)

        ckpt_path, _ = mgr.save_epoch(
            epoch=5,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"val_acc": 0.82, "val_loss": 0.65},
            config={"model_architecture": arch_name, "seed": 42},
            rng_states={},
        )

        # Load for evaluation
        eval_model, payload = load_checkpoint_for_evaluation(ckpt_path)

        assert isinstance(eval_model, nn.Module)
        assert eval_model.training is False  # Must be in eval mode
        assert payload["model_architecture"] == arch_name
        assert payload["epoch"] == 5

        # Strict weight comparison
        for p_orig, p_loaded in zip(model.parameters(), eval_model.parameters()):
            assert torch.equal(p_orig, p_loaded)


def test_load_checkpoint_missing_file_raises_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint_for_evaluation("non_existent_checkpoint_file.pt")


def test_load_checkpoint_malformed_metadata_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_ckpt = Path(tmp_dir) / "bad.pt"
        torch.save({"corrupt_key": 123}, bad_ckpt)

        with pytest.raises(ValueError, match="missing 'model_architecture'"):
            load_checkpoint_for_evaluation(bad_ckpt)


# ============================================================================
# 3. Model Evaluation Loop & Mode Verification Tests
# ============================================================================

def test_evaluate_model_uses_eval_mode_and_inference_mode() -> None:
    class ModeCheckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.was_in_eval = False
            self.was_in_inference_mode = False

        def forward(self, x):
            self.was_in_eval = not self.training
            self.was_in_inference_mode = torch.is_inference_mode_enabled()
            return self.linear(x)

    model = ModeCheckModel()
    model.train()  # Intentionally put in train mode prior to evaluate_model

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=4)

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        class_names=["cls_0", "cls_1"],
    )

    assert model.training is False
    assert model.was_in_eval is True
    assert model.was_in_inference_mode is True
    assert metrics.total_samples == 8


def test_evaluate_model_cpu_and_mps_consistency() -> None:
    model = resnet18_cifar(num_classes=10)
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    metrics_cpu = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
        class_names=CIFAR10_CLASSES,
    )

    assert metrics_cpu.total_samples == 16
    assert 0.0 <= metrics_cpu.accuracy <= 1.0

    if torch.backends.mps.is_available():
        metrics_mps = evaluate_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=torch.device("mps"),
            class_names=CIFAR10_CLASSES,
        )
        assert metrics_mps.total_samples == 16
        assert pytest.approx(metrics_cpu.accuracy, rel=1e-5) == metrics_mps.accuracy
        assert pytest.approx(metrics_cpu.loss, rel=1e-4) == metrics_mps.loss


def test_resolve_evaluation_device_behavior() -> None:
    assert resolve_evaluation_device("cpu") == torch.device("cpu")
    auto_dev = resolve_evaluation_device("auto")
    assert auto_dev.type in ("mps", "cpu")

    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS device requested"):
            resolve_evaluation_device("mps")

    with pytest.raises(ValueError, match="Unsupported evaluation device 'cuda'"):
        resolve_evaluation_device("cuda")


# ============================================================================
# 4. Atomic Serialization & Report Structure Tests
# ============================================================================

def test_save_evaluation_report_atomic_behavior() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "eval_subdir" / "test_report.json"
        data = {
            "run_id": "exp_test_01",
            "model_architecture": "resnet18_cifar",
            "metrics": {"accuracy": 0.85, "loss": 0.42},
        }

        saved_path = save_evaluation_report(data, out_file)
        assert saved_path.exists()

        # Verify no residual temporary files
        tmp_files = list(out_file.parent.glob("*.tmp*"))
        assert len(tmp_files) == 0

        # Verify JSON content
        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            assert loaded["run_id"] == "exp_test_01"
            assert loaded["metrics"]["accuracy"] == 0.85


# ============================================================================
# 5. Test-Set Isolation Verification
# ============================================================================

def test_trainer_continues_to_reject_test_loader_and_dataset_containers() -> None:
    """
    Guarantees that implementing evaluation infrastructure has not altered
    or weakened Trainer's strict test-set isolation check.
    """
    class ContainerWithTest:
        train = DataLoader(TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
        val = DataLoader(TensorDataset(torch.randn(2, 2), torch.zeros(2, dtype=torch.long)), batch_size=2)
        test = DataLoader(TensorDataset(torch.randn(2, 2), torch.zeros(2, dtype=torch.long)), batch_size=2)

    container = ContainerWithTest()
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(TypeError, match="must not be passed to Trainer"):
        Trainer(
            model=model,
            train_loader=container,  # type: ignore
            val_loader=container.val,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
        )
