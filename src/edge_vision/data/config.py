"""
Data configuration data classes and YAML loader for CIFAR-10.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    raw_dir: str = "data/raw"
    download: bool = True
    num_classes: int = 10


@dataclass
class SplitConfig:
    seed: int = 42
    train_size: int = 45000
    val_size: int = 5000
    manifest_path: str = "configs/cifar10_split_manifest.json"


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass
class AugmentationConfig:
    enabled: bool = False
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True


@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True


@dataclass
class DataPipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


def load_data_config(config_path: Optional[str] = None) -> DataPipelineConfig:
    """Load and parse YAML data configuration file."""
    if config_path is None:
        config_path = "configs/cifar10.yaml"

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    ds_data = raw_cfg.get("dataset", {})
    split_data = raw_cfg.get("split", {})
    norm_data = raw_cfg.get("normalization", {})
    aug_data = raw_cfg.get("augmentation", {})
    loader_data = raw_cfg.get("dataloader", {})

    return DataPipelineConfig(
        dataset=DatasetConfig(**ds_data),
        split=SplitConfig(**split_data),
        normalization=NormalizationConfig(**norm_data),
        augmentation=AugmentationConfig(**aug_data),
        dataloader=DataLoaderConfig(**loader_data),
    )
