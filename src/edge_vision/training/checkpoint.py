"""
Robust, atomic checkpoint management, serialization, and resume validation.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision


class CheckpointManager:
    """
    Manages atomic persistence of model checkpoints, tracking latest and best-validation checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        metric_for_best: str = "val_acc",
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_for_best = metric_for_best
        self.mode = mode.lower()

        if self.mode not in ("max", "min"):
            raise ValueError(f"Checkpoint mode must be 'max' or 'min', got '{self.mode}'")

        self.best_metric_value: float = (
            float("-inf") if self.mode == "max" else float("inf")
        )
        self.best_epoch: int = 0
        self.best_checkpoint_path: Path = self.checkpoint_dir / "best_checkpoint.pt"
        self.latest_checkpoint_path: Path = self.checkpoint_dir / "latest_checkpoint.pt"

    def is_better(self, val_acc: float, val_loss: float, best_val_acc: float, best_val_loss: float) -> bool:
        """
        Determines whether a candidate epoch improves on the best record:
        1. Higher validation accuracy wins.
        2. If accuracy ties (within 1e-6), lower validation loss wins.
        """
        if abs(val_acc - best_val_acc) > 1e-6:
            return val_acc > best_val_acc
        return val_loss < best_val_loss

    def _save_atomic(self, payload: Dict[str, Any], destination: Path) -> None:
        """
        Atomically saves payload by writing to a process-unique temporary file, then renaming.
        """
        temp_path = destination.with_suffix(f".tmp.{os.getpid()}")
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, destination)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

    def save_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        rng_states: Dict[str, Any],
        dataset_manifest_checksum: Optional[str] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Persists latest checkpoint and conditionally updates best checkpoint.

        Parameters:
            epoch: Current completed epoch index.
            model: Active model instance.
            optimizer: Active optimizer.
            scheduler: Active LR scheduler (or None).
            metrics: Dictionary containing epoch metrics (train_loss, train_acc, val_loss, val_acc, etc.).
            config: Serialized training configuration dictionary.
            rng_states: Captured RNG states across runtimes and DataLoader generator.
            dataset_manifest_checksum: Optional split manifest SHA-256 checksum.

        Returns:
            Tuple of (latest_checkpoint_path, best_checkpoint_path_if_updated_else_None).
        """
        val_acc = metrics.get("val_acc", 0.0)
        val_loss = metrics.get("val_loss", float("inf"))

        is_new_best = False
        if epoch == 1:
            is_new_best = True
            self.best_metric_value = val_acc
            self._best_val_loss = val_loss
            self.best_epoch = epoch
        elif self.is_better(val_acc, val_loss, self.best_metric_value, getattr(self, "_best_val_loss", float("inf"))):
            is_new_best = True
            self.best_metric_value = val_acc
            self._best_val_loss = val_loss
            self.best_epoch = epoch

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_architecture": config.get("model_architecture", "unknown"),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": config,
            "seed": config.get("seed", 42),
            "run_id": config.get("experiment_id", "unnamed"),
            "dataset_manifest_checksum": dataset_manifest_checksum,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rng_states": rng_states,
        }

        # Save latest checkpoint atomically
        self._save_atomic(payload, self.latest_checkpoint_path)

        saved_best: Optional[Path] = None
        if is_new_best:
            self._save_atomic(payload, self.best_checkpoint_path)
            saved_best = self.best_checkpoint_path

        return self.latest_checkpoint_path, saved_best

    @staticmethod
    def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads a checkpoint dictionary from disk.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc

        return payload

    @staticmethod
    def validate_checkpoint_compatibility(
        checkpoint_payload: Dict[str, Any],
        expected_config: Dict[str, Any],
    ) -> None:
        """
        Validates that a checkpoint is strictly compatible with the intended resuming experiment.
        """
        required_fields = [
            "epoch",
            "model_architecture",
            "model_state_dict",
            "optimizer_state_dict",
            "config",
            "rng_states",
        ]
        for f in required_fields:
            if f not in checkpoint_payload:
                raise ValueError(f"Malformed checkpoint: missing required field '{f}'")

        ckpt_arch = checkpoint_payload.get("model_architecture")
        exp_arch = expected_config.get("model_architecture")
        if ckpt_arch != exp_arch:
            raise ValueError(
                f"Architecture mismatch: checkpoint was created with '{ckpt_arch}', "
                f"but current configuration specifies '{exp_arch}'"
            )

        ckpt_seed = checkpoint_payload.get("seed")
        exp_seed = expected_config.get("seed")
        if ckpt_seed != exp_seed:
            raise ValueError(
                f"Seed mismatch: checkpoint was created with seed {ckpt_seed}, "
                f"but current configuration specifies seed {exp_seed}"
            )
