"""
Experiment run tracking, structured metrics recording (CSV), and run summary (JSON).
"""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Optional, Union
import psutil
import torch
import torchvision

from edge_vision.training.metrics import EpochMetrics


class ExperimentLogger:
    """
    Handles file-based experiment logging, recording per-epoch tabular metrics to CSV
    and finalized experiment summaries to JSON.
    """

    CSV_HEADERS = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "learning_rate",
        "epoch_duration_sec",
        "samples_per_sec",
        "peak_rss_mb",
    ]

    def __init__(self, experiment_dir: Union[str, Path], experiment_id: str) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.csv_path = self.experiment_dir / "metrics_history.csv"
        self.summary_path = self.experiment_dir / "experiment_summary.json"

        self._start_time_iso = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()

        # Initialize CSV if not already present
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(self.CSV_HEADERS)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        learning_rate: float,
        peak_rss_mb: Optional[float] = None,
    ) -> None:
        """
        Appends an epoch record to metrics_history.csv.
        """
        if peak_rss_mb is None:
            proc = psutil.Process(os.getpid())
            peak_rss_mb = proc.memory_info().rss / 1_000_000

        row = [
            epoch,
            f"{train_metrics.loss:.6f}",
            f"{train_metrics.accuracy:.6f}",
            f"{val_metrics.loss:.6f}",
            f"{val_metrics.accuracy:.6f}",
            f"{learning_rate:.6e}",
            f"{train_metrics.duration_seconds + val_metrics.duration_seconds:.2f}",
            f"{train_metrics.samples_per_second:.1f}",
            f"{peak_rss_mb:.2f}",
        ]

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(row)

    def write_summary(
        self,
        config_dict: Dict[str, Any],
        best_epoch: int,
        best_val_acc: float,
        best_val_loss: float,
        final_train_loss: float,
        final_train_acc: float,
        final_val_loss: float,
        final_val_acc: float,
        device_str: str,
        status: str = "completed",
        dataset_manifest_checksum: Optional[str] = None,
    ) -> None:
        """
        Writes atomic experiment summary JSON.
        """
        end_time_iso = datetime.now(timezone.utc).isoformat()
        total_duration = time.perf_counter() - self._start_perf

        summary: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "status": status,
            "model_architecture": config_dict.get("model_architecture"),
            "seed": config_dict.get("seed"),
            "device": device_str,
            "timestamps": {
                "start": self._start_time_iso,
                "end": end_time_iso,
                "total_duration_seconds": round(total_duration, 2),
            },
            "best_metrics": {
                "best_epoch": best_epoch,
                "best_val_acc": round(best_val_acc, 6),
                "best_val_loss": round(best_val_loss, 6),
            },
            "final_metrics": {
                "train_loss": round(final_train_loss, 6),
                "train_acc": round(final_train_acc, 6),
                "val_loss": round(final_val_loss, 6),
                "val_acc": round(final_val_acc, 6),
            },
            "configuration": config_dict,
            "dataset_manifest_checksum": dataset_manifest_checksum,
            "environment": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "checkpoints": {
                "best_checkpoint": str(self.experiment_dir / "best_checkpoint.pt"),
                "latest_checkpoint": str(self.experiment_dir / "latest_checkpoint.pt"),
            },
        }

        # Atomic write
        temp_path = self.summary_path.with_suffix(f".tmp.{os.getpid()}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            os.replace(temp_path, self.summary_path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
