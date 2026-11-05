"""
Configuration data classes and YAML loader for synthetic image corruptions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union
import yaml


@dataclass
class SingleCorruptionConfig:
    description: str
    parameter_name: str
    is_stochastic: bool
    severities: Dict[int, Union[int, float]]


@dataclass
class CorruptionSuiteConfig:
    default_seed: int = 42
    corruptions: Dict[str, SingleCorruptionConfig] = field(default_factory=dict)


def get_default_corruption_config_path() -> Path:
    """
    Resolves the canonical path to configs/corruptions.yaml.
    Guarantees resolution succeeds regardless of the current working directory.
    """
    # 1. Resolve relative to repo root (three directory levels above src/edge_vision/corruptions/config.py)
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "configs" / "corruptions.yaml"
    if candidate.is_file():
        return candidate

    # 2. Fall back to current working directory if repo root is the current working directory
    cwd_candidate = Path("configs/corruptions.yaml").resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    return candidate


def load_corruption_config(config_path: Optional[Union[str, Path]] = None) -> CorruptionSuiteConfig:
    """
    Loads and validates the corruption suite configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, resolves the canonical path
                     to 'configs/corruptions.yaml'.

    Returns:
        Populated CorruptionSuiteConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If configuration values are invalid or severities 1-5 are missing.
    """
    if config_path is None:
        path = get_default_corruption_config_path()
    else:
        path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Corruption config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    default_seed = int(raw_cfg.get("default_seed", 42))
    raw_corruptions = raw_cfg.get("corruptions", {})

    if not raw_corruptions:
        raise ValueError(f"No corruptions defined in {path}")

    corruptions: Dict[str, SingleCorruptionConfig] = {}
    for name, c_data in raw_corruptions.items():
        severities = {int(k): v for k, v in c_data.get("severities", {}).items()}
        missing = set(range(1, 6)) - set(severities.keys())
        if missing:
            raise ValueError(f"Corruption '{name}' is missing required severity levels: {sorted(missing)}")

        corruptions[name] = SingleCorruptionConfig(
            description=c_data.get("description", ""),
            parameter_name=c_data.get("parameter_name", "param"),
            is_stochastic=bool(c_data.get("is_stochastic", False)),
            severities=severities,
        )

    return CorruptionSuiteConfig(
        default_seed=default_seed,
        corruptions=corruptions,
    )
