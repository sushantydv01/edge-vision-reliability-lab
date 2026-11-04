"""
Type-safe training configuration schemas and validation for CIFAR-10 experiments.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from edge_vision.models.factory import MODEL_REGISTRY


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    learning_rate: float = 0.05
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0005


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    t_max: int = 100
    eta_min: float = 0.0001
    warmup_epochs: int = 0


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_latest: bool = True
    metric_for_best: str = "val_acc"
    mode: str = "max"


@dataclass
class AugmentationTrainConfig:
    enabled: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True


@dataclass
class DataLoaderTrainConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainingConfig:
    experiment_id: str = "exp_baseline_resnet18"
    model_architecture: str = "resnet18_cifar"
    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    loss_function: str = "cross_entropy"
    max_grad_norm: Optional[float] = None
    deterministic: bool = False
    data_config_path: str = "configs/cifar10.yaml"
    output_dir: str = "experiments"

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    augmentation: AugmentationTrainConfig = field(default_factory=AugmentationTrainConfig)
    dataloader: DataLoaderTrainConfig = field(default_factory=DataLoaderTrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a deterministic dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to formatted JSON string with sorted keys."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def validate_training_config(config: TrainingConfig) -> None:
    """
    Validates configuration values and raises ValueError with clear diagnostics if invalid.
    """
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")

    if config.dataloader.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.dataloader.batch_size}")

    if config.seed < 0:
        raise ValueError(f"seed must be non-negative, got {config.seed}")

    valid_devices = ("auto", "cpu", "mps")
    if config.device not in valid_devices:
        raise ValueError(f"device '{config.device}' not supported. Expected one of {valid_devices}")

    if config.model_architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"model_architecture '{config.model_architecture}' not registered. "
            f"Available: {sorted(list(MODEL_REGISTRY.keys()))}"
        )

    valid_optimizers = ("sgd", "adamw")
    if config.optimizer.name.lower() not in valid_optimizers:
        raise ValueError(
            f"optimizer '{config.optimizer.name}' not supported. Expected one of {valid_optimizers}"
        )

    if config.optimizer.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.optimizer.learning_rate}")

    if config.optimizer.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {config.optimizer.weight_decay}")

    if config.optimizer.momentum < 0:
        raise ValueError(f"momentum must be non-negative, got {config.optimizer.momentum}")

    valid_schedulers = ("cosine", "step", "none")
    if config.scheduler.name.lower() not in valid_schedulers:
        raise ValueError(
            f"scheduler '{config.scheduler.name}' not supported. Expected one of {valid_schedulers}"
        )

    if config.scheduler.name.lower() == "cosine":
        if config.scheduler.t_max <= 0:
            raise ValueError(f"t_max must be positive for cosine scheduler, got {config.scheduler.t_max}")
        if config.scheduler.eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {config.scheduler.eta_min}")

    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive when set, got {config.max_grad_norm}")


def load_training_config(config_path: Optional[str] = None) -> TrainingConfig:
    """
    Load and parse a YAML training configuration file.
    """
    if config_path is None:
        config_path = "configs/train_baseline.yaml"

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    opt_data = raw_cfg.get("optimizer", {})
    sched_data = raw_cfg.get("scheduler", {})
    ckpt_data = raw_cfg.get("checkpoint", {})
    aug_data = raw_cfg.get("augmentation", {})
    dl_data = raw_cfg.get("dataloader", {})

    top_level_keys = {
        "experiment_id",
        "model_architecture",
        "seed",
        "device",
        "epochs",
        "loss_function",
        "max_grad_norm",
        "deterministic",
        "data_config_path",
        "output_dir",
    }
    kwargs = {k: raw_cfg[k] for k in top_level_keys if k in raw_cfg}

    cfg = TrainingConfig(
        optimizer=OptimizerConfig(**opt_data),
        scheduler=SchedulerConfig(**sched_data),
        checkpoint=CheckpointConfig(**ckpt_data),
        augmentation=AugmentationTrainConfig(**aug_data),
        dataloader=DataLoaderTrainConfig(**dl_data),
        **kwargs,
    )
    validate_training_config(cfg)
    return cfg
