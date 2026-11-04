"""
Metric calculation, sample-weighted loss aggregation, and epoch telemetry.
"""

from dataclasses import dataclass
import time
from typing import Optional
import torch


@dataclass
class EpochMetrics:
    """
    Structured metrics calculated over a complete training or evaluation epoch.
    """
    loss: float
    accuracy: float
    total_samples: int
    duration_seconds: float
    samples_per_second: float

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "total_samples": self.total_samples,
            "duration_seconds": self.duration_seconds,
            "samples_per_second": self.samples_per_second,
        }


class MetricTracker:
    """
    Accumulator for batch-level metrics, performing exact sample-weighted loss aggregation
    and top-1 classification accuracy computation.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all counters and timers."""
        self._running_weighted_loss: float = 0.0
        self._running_correct: int = 0
        self._total_samples: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start_timer(self) -> None:
        """Start the epoch timer."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop the epoch timer."""
        self._end_time = time.perf_counter()

    def update(
        self,
        batch_loss: float,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Update accumulator with batch results.

        Parameters:
            batch_loss: Average scalar loss for the batch.
            preds: Predicted class indices or raw logits tensor (shape: [B] or [B, C]).
            targets: Ground-truth target class labels tensor (shape: [B]).
            batch_size: Explicit batch size. If None, inferred from targets.size(0).
        """
        if batch_size is None:
            batch_size = targets.size(0)

        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)

        correct = (preds == targets).sum().item()

        self._running_weighted_loss += float(batch_loss) * batch_size
        self._running_correct += int(correct)
        self._total_samples += int(batch_size)

    def compute(self) -> EpochMetrics:
        """
        Computes final epoch metrics using sample-weighted mean loss and exact top-1 accuracy.

        Returns:
            EpochMetrics dataclass.
        """
        if self._total_samples == 0:
            return EpochMetrics(
                loss=0.0,
                accuracy=0.0,
                total_samples=0,
                duration_seconds=0.0,
                samples_per_second=0.0,
            )

        mean_loss = self._running_weighted_loss / self._total_samples
        accuracy = self._running_correct / self._total_samples

        duration = 0.0
        if self._start_time is not None:
            end_t = self._end_time if self._end_time is not None else time.perf_counter()
            duration = max(end_t - self._start_time, 1e-6)

        throughput = self._total_samples / duration if duration > 0 else 0.0

        return EpochMetrics(
            loss=mean_loss,
            accuracy=accuracy,
            total_samples=self._total_samples,
            duration_seconds=duration,
            samples_per_second=throughput,
        )
