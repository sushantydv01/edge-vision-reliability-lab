"""
Trainer class orchestrating training loops, validation, optimization, and checkpointing.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_vision.training.checkpoint import CheckpointManager
from edge_vision.training.config import TrainingConfig
from edge_vision.training.experiment import ExperimentLogger
from edge_vision.training.metrics import EpochMetrics, MetricTracker
from edge_vision.training.seeding import get_rng_states


class Trainer:
    """
    Standard FP32 training orchestrator for CIFAR-10 classification models.

    Strictly enforces test-set isolation: only train_loader and val_loader may be provided.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[TrainingConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        device: Optional[Union[str, torch.device]] = None,
        loader_generator: Optional[torch.Generator] = None,
    ) -> None:
        # API-level enforcement of test-set isolation
        if hasattr(train_loader, "test") or hasattr(val_loader, "test"):
            raise TypeError("Dataset container containing test data must not be passed to Trainer.")

        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader):
            raise TypeError("train_loader and val_loader must be torch.utils.data.DataLoader instances.")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config if config is not None else TrainingConfig()
        self.checkpoint_manager = checkpoint_manager
        self.experiment_logger = experiment_logger
        self.loader_generator = loader_generator

        # Device resolution
        self.device = self._resolve_device(device, self.config.device)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(
        explicit_device: Optional[Union[str, torch.device]],
        config_device: str,
    ) -> torch.device:
        """
        Resolves the execution device. Raises RuntimeError if an unavailable device is requested.
        """
        req = explicit_device if explicit_device is not None else config_device
        if isinstance(req, torch.device):
            return req

        req_str = str(req).lower()
        if req_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif req_str == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS device requested but torch.backends.mps.is_available() is False.")
            return torch.device("mps")
        elif req_str == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device string '{req_str}'. Expected 'auto', 'cpu', or 'mps'.")

    def _sync_device(self) -> None:
        """Synchronize accelerator device for accurate boundary timing."""
        if self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif self.device.type == "cuda" and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        Executes one full training epoch over train_loader.
        """
        self.model.train()
        tracker = MetricTracker()

        self._sync_device()
        tracker.start_timer()

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss ({loss.item()}) encountered during training epoch {epoch}"
                )

            loss.backward()

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()
            tracker.update(loss.item(), outputs.detach(), targets, batch_size=inputs.size(0))

        self._sync_device()
        tracker.stop_timer()

        return tracker.compute()

    def validate_epoch(self, epoch: int) -> EpochMetrics:
        """
        Executes one full validation epoch over val_loader without updating model weights.
        """
        self.model.eval()
        tracker = MetricTracker()

        self._sync_device()
        tracker.start_timer()

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                tracker.update(loss.item(), outputs, targets, batch_size=inputs.size(0))

        self._sync_device()
        tracker.stop_timer()

        return tracker.compute()

    def fit(
        self,
        epochs: Optional[int] = None,
        start_epoch: int = 1,
    ) -> Dict[str, Any]:
        """
        Orchestrates training, validation, scheduling, checkpointing, and experiment logging.

        Parameters:
            epochs: Total epochs to train (defaults to config.epochs).
            start_epoch: Epoch to start at (defaults to 1; >1 when resuming).

        Returns:
            Dictionary containing run results and metrics summary.
        """
        total_epochs = epochs if epochs is not None else self.config.epochs

        last_train_metrics: Optional[EpochMetrics] = None
        last_val_metrics: Optional[EpochMetrics] = None

        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            last_train_metrics = train_metrics
            last_val_metrics = val_metrics

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler once per completed validation epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint atomically if manager is configured
            if self.checkpoint_manager is not None:
                rng_states = get_rng_states(loader_generator=self.loader_generator)
                metrics_dict = {
                    "train_loss": train_metrics.loss,
                    "train_acc": train_metrics.accuracy,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                }
                self.checkpoint_manager.save_epoch(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=metrics_dict,
                    config=self.config.to_dict(),
                    rng_states=rng_states,
                )

            # Record epoch in experiment logger
            if self.experiment_logger is not None:
                self.experiment_logger.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                )

        # Write finalized experiment summary
        best_epoch = self.checkpoint_manager.best_epoch if self.checkpoint_manager else total_epochs
        best_val_acc = (
            self.checkpoint_manager.best_metric_value
            if self.checkpoint_manager
            else (last_val_metrics.accuracy if last_val_metrics else 0.0)
        )
        best_val_loss = (
            getattr(self.checkpoint_manager, "_best_val_loss", 0.0)
            if self.checkpoint_manager
            else (last_val_metrics.loss if last_val_metrics else 0.0)
        )

        final_train_loss = last_train_metrics.loss if last_train_metrics else 0.0
        final_train_acc = last_train_metrics.accuracy if last_train_metrics else 0.0
        final_val_loss = last_val_metrics.loss if last_val_metrics else 0.0
        final_val_acc = last_val_metrics.accuracy if last_val_metrics else 0.0

        if self.experiment_logger is not None:
            self.experiment_logger.write_summary(
                config_dict=self.config.to_dict(),
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                final_train_loss=final_train_loss,
                final_train_acc=final_train_acc,
                final_val_loss=final_val_loss,
                final_val_acc=final_val_acc,
                device_str=str(self.device),
                status="completed",
            )

        return {
            "total_epochs": total_epochs,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        }
