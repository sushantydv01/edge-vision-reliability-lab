"""
Comprehensive unit and integration tests for the training engine, configuration,
reproducibility, checkpointing, metrics, and data integration.
"""

import copy
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from edge_vision.models import resnet18_cifar
from edge_vision.training.config import (
    TrainingConfig,
    load_training_config,
    validate_training_config,
)
from edge_vision.training.seeding import (
    set_seed,
    get_rng_states,
    set_rng_states,
)
from edge_vision.training.metrics import (
    MetricTracker,
    EpochMetrics,
)
from edge_vision.training.checkpoint import (
    CheckpointManager,
)
from edge_vision.training.experiment import (
    ExperimentLogger,
)
from edge_vision.training.data import (
    get_training_dataloaders,
)
from edge_vision.training.trainer import (
    Trainer,
)


# ============================================================================
# 1. Configuration Tests
# ============================================================================

def test_valid_configuration_loads() -> None:
    cfg = load_training_config("configs/train_baseline.yaml")
    assert cfg.epochs == 100
    assert cfg.dataloader.batch_size == 64
    assert cfg.optimizer.name == "sgd"
    assert cfg.optimizer.learning_rate == 0.05
    assert cfg.optimizer.momentum == 0.9
    assert cfg.optimizer.weight_decay == 0.0005
    assert cfg.scheduler.name == "cosine"
    assert cfg.augmentation.enabled is True
    assert cfg.seed == 42


def test_invalid_epochs_rejected() -> None:
    cfg = TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="epochs must be positive"):
        validate_training_config(cfg)


def test_invalid_batch_size_rejected() -> None:
    cfg = TrainingConfig()
    cfg.dataloader.batch_size = -8
    with pytest.raises(ValueError, match="batch_size must be positive"):
        validate_training_config(cfg)


def test_invalid_learning_rate_rejected() -> None:
    cfg = TrainingConfig()
    cfg.optimizer.learning_rate = 0.0
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        validate_training_config(cfg)


def test_invalid_weight_decay_rejected() -> None:
    cfg = TrainingConfig()
    cfg.optimizer.weight_decay = -0.001
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        validate_training_config(cfg)


def test_invalid_scheduler_values_rejected() -> None:
    cfg = TrainingConfig()
    cfg.scheduler.t_max = 0
    with pytest.raises(ValueError, match="t_max must be positive for cosine scheduler"):
        validate_training_config(cfg)


def test_unsupported_device_rejected() -> None:
    cfg = TrainingConfig(device="invalid_dev")
    with pytest.raises(ValueError, match="device 'invalid_dev' not supported"):
        validate_training_config(cfg)


def test_deterministic_serialization() -> None:
    cfg1 = load_training_config("configs/train_baseline.yaml")
    json1 = cfg1.to_json()
    cfg2 = load_training_config("configs/train_baseline.yaml")
    json2 = cfg2.to_json()
    assert json1 == json2
    loaded = json.loads(json1)
    assert loaded["optimizer"]["learning_rate"] == 0.05


# ============================================================================
# 2. Reproducibility Tests
# ============================================================================

def test_set_seed_behavior() -> None:
    set_seed(1234)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.randn(5)

    set_seed(1234)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.randn(5)

    assert r1 == r2
    assert np.array_equal(np1, np2)
    assert torch.equal(t1, t2)


def test_dataloader_generator_state_capture_and_restoration() -> None:
    dataset = TensorDataset(torch.arange(40))
    gen1 = torch.Generator().manual_seed(99)
    loader1 = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen1)

    # Consume epoch 1
    _ = [b[0].tolist() for b in loader1]
    saved_gen_state = gen1.get_state()

    # Epoch 2 uninterrupted
    epoch2_uninterrupted = [b[0].tolist() for b in loader1]

    # Resume scenario
    gen2 = torch.Generator().manual_seed(99)
    loader2 = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen2)
    _ = [b[0].tolist() for b in loader2]

    # Restore generator state
    gen2.set_state(saved_gen_state)
    epoch2_resumed = [b[0].tolist() for b in loader2]

    assert epoch2_uninterrupted == epoch2_resumed


def test_rng_states_roundtrip() -> None:
    gen = torch.Generator().manual_seed(42)
    states = get_rng_states(loader_generator=gen)

    # Advance generators
    _ = random.random()
    _ = np.random.rand()
    _ = torch.randn(1)
    _ = torch.randint(0, 100, (1,), generator=gen)

    # Restore
    set_rng_states(states, loader_generator=gen)

    # Next values should match fresh state after save
    set_rng_states(states, loader_generator=gen)
    t_after_restore = torch.randn(5)

    set_rng_states(states, loader_generator=gen)
    t_repeat = torch.randn(5)

    assert torch.equal(t_after_restore, t_repeat)


# ============================================================================
# 3. Metrics Tests
# ============================================================================

def test_sample_weighted_loss_aggregation() -> None:
    tracker = MetricTracker()
    # Batch 1: 10 samples, loss 1.0
    preds1 = torch.tensor([0] * 10)
    targets1 = torch.tensor([0] * 10)
    tracker.update(batch_loss=1.0, preds=preds1, targets=targets1, batch_size=10)

    # Batch 2: 2 samples, loss 2.0 (unequal final batch)
    preds2 = torch.tensor([0, 1])
    targets2 = torch.tensor([0, 0])
    tracker.update(batch_loss=2.0, preds=preds2, targets=targets2, batch_size=2)

    metrics = tracker.compute()
    expected_weighted_loss = (10 * 1.0 + 2 * 2.0) / 12  # 14 / 12 = 1.166667
    assert pytest.approx(metrics.loss, rel=1e-5) == expected_weighted_loss

    # Accuracy: (10 + 1) / 12 = 11 / 12
    expected_acc = 11 / 12
    assert pytest.approx(metrics.accuracy, rel=1e-5) == expected_acc
    assert metrics.total_samples == 12


def test_empty_metric_tracker() -> None:
    tracker = MetricTracker()
    metrics = tracker.compute()
    assert metrics.total_samples == 0
    assert metrics.loss == 0.0
    assert metrics.accuracy == 0.0


# ============================================================================
# 4. Trainer Tests & Test-Set Isolation
# ============================================================================

def test_trainer_rejects_test_loader_or_dataset_container() -> None:
    class FakeDatasetContainer:
        train = DataLoader(TensorDataset(torch.randn(10, 3)), batch_size=2)
        val = DataLoader(TensorDataset(torch.randn(5, 3)), batch_size=2)
        test = DataLoader(TensorDataset(torch.randn(5, 3)), batch_size=2)

    fake_container = FakeDatasetContainer()
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Passing an object having a .test attribute must raise TypeError
    with pytest.raises(TypeError, match="must not be passed to Trainer"):
        Trainer(
            model=model,
            train_loader=fake_container,  # type: ignore
            val_loader=fake_container.val,
            criterion=criterion,
            optimizer=optimizer,
        )


def test_trainer_train_and_val_epoch_execution_synthetic() -> None:
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    initial_weight = model[0].weight.clone()

    x_train = torch.randn(32, 8)
    y_train = torch.randint(0, 2, (32,))
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    x_val = torch.randn(16, 8)
    y_val = torch.randint(0, 2, (16,))
    val_ds = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    # 1. Validation before training does not update weights
    val_metrics = trainer.validate_epoch(1)
    assert torch.equal(model[0].weight, initial_weight)
    assert 0.0 <= val_metrics.accuracy <= 1.0

    # 2. Training updates weights
    train_metrics = trainer.train_epoch(1)
    assert not torch.equal(model[0].weight, initial_weight)
    assert train_metrics.total_samples == 32
    assert train_metrics.duration_seconds > 0


def test_trainer_non_finite_loss_raises_floating_point_error() -> None:
    model = nn.Linear(4, 2)
    # Target loss returning NaN
    class NaNCriterion(nn.Module):
        def forward(self, o, t):
            return torch.tensor(float("nan"), requires_grad=True)

    train_loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.zeros(4, dtype=torch.long)), batch_size=2)
    val_loader = DataLoader(TensorDataset(torch.randn(2, 4), torch.zeros(2, dtype=torch.long)), batch_size=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=NaNCriterion(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
    )

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        trainer.train_epoch(1)


# ============================================================================
# 5. Checkpointing & Atomic Saves Tests
# ============================================================================

def test_atomic_checkpoint_saving_and_loading() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(tmp_dir)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        metrics_ep1 = {"train_loss": 1.5, "train_acc": 0.4, "val_loss": 1.4, "val_acc": 0.5}
        config = {"model_architecture": "linear_test", "seed": 42}
        rng = {"cpu": "dummy"}

        latest_path, best_path = manager.save_epoch(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics=metrics_ep1,
            config=config,
            rng_states=rng,
        )

        assert latest_path.exists()
        assert best_path is not None and best_path.exists()

        # Check no dangling temporary files
        temp_files = list(Path(tmp_dir).glob("*.tmp*"))
        assert len(temp_files) == 0

        # Load and verify
        payload = manager.load_checkpoint(latest_path)
        assert payload["epoch"] == 1
        assert payload["model_architecture"] == "linear_test"
        assert payload["best_metric_value"] == 0.5


def test_checkpoint_selection_logic() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = CheckpointManager(tmp_dir)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = {"model_architecture": "test", "seed": 42}

        # Epoch 1: val_acc 0.50
        _, b1 = manager.save_epoch(1, model, optimizer, None, {"val_acc": 0.50, "val_loss": 1.2}, config, {})
        assert b1 is not None

        # Epoch 2: val_acc 0.45 (worse -> best not updated)
        _, b2 = manager.save_epoch(2, model, optimizer, None, {"val_acc": 0.45, "val_loss": 1.1}, config, {})
        assert b2 is None
        assert manager.best_epoch == 1

        # Epoch 3: val_acc 0.50, lower val_loss 1.0 (tie-break win -> best updated)
        _, b3 = manager.save_epoch(3, model, optimizer, None, {"val_acc": 0.50, "val_loss": 1.0}, config, {})
        assert b3 is not None
        assert manager.best_epoch == 3

        # Epoch 4: val_acc 0.55 (better -> best updated)
        _, b4 = manager.save_epoch(4, model, optimizer, None, {"val_acc": 0.55, "val_loss": 1.0}, config, {})
        assert b4 is not None
        assert manager.best_epoch == 4


def test_checkpoint_compatibility_validation() -> None:
    payload = {
        "epoch": 5,
        "model_architecture": "resnet18_cifar",
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "config": {"seed": 42},
        "seed": 42,
        "rng_states": {},
    }

    # Matching config
    CheckpointManager.validate_checkpoint_compatibility(payload, {"model_architecture": "resnet18_cifar", "seed": 42})

    # Architecture mismatch
    with pytest.raises(ValueError, match="Architecture mismatch"):
        CheckpointManager.validate_checkpoint_compatibility(payload, {"model_architecture": "mobilenet_v3_small_cifar", "seed": 42})

    # Seed mismatch
    with pytest.raises(ValueError, match="Seed mismatch"):
        CheckpointManager.validate_checkpoint_compatibility(payload, {"model_architecture": "resnet18_cifar", "seed": 999})


# ============================================================================
# 6. Resume-Equivalence Test
# ============================================================================

def test_epoch_boundary_resume_equivalence() -> None:
    """
    Verifies that interrupted 1-epoch training + resume from checkpoint yields
    the identical model state and metrics as uninterrupted 2-epoch training.
    """
    torch.manual_seed(42)
    x = torch.randn(64, 4)
    y = torch.randint(0, 2, (64,))

    dataset = TensorDataset(x, y)

    # Common initial model
    init_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    initial_state = copy.deepcopy(init_model.state_dict())

    # --- Run A: Uninterrupted 2 Epochs ---
    set_seed(42)
    model_a = copy.deepcopy(init_model)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05, momentum=0.9)
    crit_a = nn.CrossEntropyLoss()

    gen_a = torch.Generator().manual_seed(42)
    loader_a = DataLoader(dataset, batch_size=16, shuffle=True, generator=gen_a)

    trainer_a = Trainer(
        model=model_a,
        train_loader=loader_a,
        val_loader=loader_a,
        criterion=crit_a,
        optimizer=opt_a,
        device=torch.device("cpu"),
        loader_generator=gen_a,
    )

    metrics_a1 = trainer_a.train_epoch(1)
    metrics_a2 = trainer_a.train_epoch(2)

    # --- Run B: 1 Epoch -> Save Checkpoint -> Resume Epoch 2 ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        set_seed(42)
        model_b = copy.deepcopy(init_model)
        model_b.load_state_dict(initial_state)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05, momentum=0.9)
        crit_b = nn.CrossEntropyLoss()

        gen_b = torch.Generator().manual_seed(42)
        loader_b = DataLoader(dataset, batch_size=16, shuffle=True, generator=gen_b)

        ckpt_mgr = CheckpointManager(tmp_dir)

        trainer_b1 = Trainer(
            model=model_b,
            train_loader=loader_b,
            val_loader=loader_b,
            criterion=crit_b,
            optimizer=opt_b,
            device=torch.device("cpu"),
            loader_generator=gen_b,
            checkpoint_manager=ckpt_mgr,
        )

        metrics_b1 = trainer_b1.train_epoch(1)
        assert pytest.approx(metrics_a1.loss, rel=1e-5) == metrics_b1.loss

        # Save checkpoint at epoch 1 boundary
        rng_states_b1 = get_rng_states(loader_generator=gen_b)
        ckpt_path, _ = ckpt_mgr.save_epoch(
            epoch=1,
            model=model_b,
            optimizer=opt_b,
            scheduler=None,
            metrics={"train_loss": metrics_b1.loss, "val_loss": 0.0, "val_acc": 0.0},
            config={"model_architecture": "test", "seed": 42},
            rng_states=rng_states_b1,
        )

        # Restore into fresh model and trainer
        ckpt = CheckpointManager.load_checkpoint(ckpt_path)
        model_resumed = copy.deepcopy(init_model)
        model_resumed.load_state_dict(ckpt["model_state_dict"])
        opt_resumed = torch.optim.SGD(model_resumed.parameters(), lr=0.05, momentum=0.9)
        opt_resumed.load_state_dict(ckpt["optimizer_state_dict"])

        gen_resumed = torch.Generator().manual_seed(42)
        loader_resumed = DataLoader(dataset, batch_size=16, shuffle=True, generator=gen_resumed)

        # Restore RNG states including DataLoader generator
        set_rng_states(ckpt["rng_states"], loader_generator=gen_resumed)

        trainer_resumed = Trainer(
            model=model_resumed,
            train_loader=loader_resumed,
            val_loader=loader_resumed,
            criterion=nn.CrossEntropyLoss(),
            optimizer=opt_resumed,
            device=torch.device("cpu"),
            loader_generator=gen_resumed,
        )

        metrics_b2 = trainer_resumed.train_epoch(2)

        # Exact loss match
        assert pytest.approx(metrics_a2.loss, rel=1e-5) == metrics_b2.loss

        # Exact parameter equality
        for p_a, p_b in zip(model_a.parameters(), model_resumed.parameters()):
            assert torch.equal(p_a, p_b)


# ============================================================================
# 7. Experiment Logger Tests
# ============================================================================

def test_experiment_logger_writes_csv_and_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ExperimentLogger(tmp_dir, experiment_id="exp_test_run")

        m1 = EpochMetrics(loss=1.2, accuracy=0.6, total_samples=100, duration_seconds=1.5, samples_per_second=66.6)
        m2 = EpochMetrics(loss=1.1, accuracy=0.65, total_samples=50, duration_seconds=0.5, samples_per_second=100.0)

        logger.log_epoch(epoch=1, train_metrics=m1, val_metrics=m2, learning_rate=0.05)

        assert logger.csv_path.exists()
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2  # header + 1 data line
            assert "epoch" in lines[0]
            assert "1.200000" in lines[1]

        logger.write_summary(
            config_dict={"model_architecture": "test"},
            best_epoch=1,
            best_val_acc=0.65,
            best_val_loss=1.1,
            final_train_loss=1.2,
            final_train_acc=0.6,
            final_val_loss=1.1,
            final_val_acc=0.65,
            device_str="cpu",
        )

        assert logger.summary_path.exists()
        with open(logger.summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["experiment_id"] == "exp_test_run"
            assert data["best_metrics"]["best_val_acc"] == 0.65


# ============================================================================
# 8. Data Integration Tests
# ============================================================================

def test_get_training_dataloaders_augmentation_toggle() -> None:
    cfg = load_training_config("configs/train_baseline.yaml")
    cfg.augmentation.enabled = True
    train_loader, val_loader, generator = get_training_dataloaders(cfg)

    # 1. Dataset lengths
    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000

    # 2. Augmentation applied only to training transform
    train_transforms = train_loader.dataset.transform.transforms
    train_transform_types = [type(t).__name__ for t in train_transforms]
    assert "RandomCrop" in train_transform_types
    assert "RandomHorizontalFlip" in train_transform_types

    # 3. Validation transform is unaugmented
    val_transforms = val_loader.dataset.transform.transforms
    val_transform_types = [type(t).__name__ for t in val_transforms]
    assert "RandomCrop" not in val_transform_types
    assert "RandomHorizontalFlip" not in val_transform_types

    # 4. Manifest path unchanged and exists
    assert Path(cfg.data_config_path).exists()
