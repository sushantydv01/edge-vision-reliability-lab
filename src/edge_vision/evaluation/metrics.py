"""
Evaluation metrics calculation, sample-weighted loss aggregation, and confusion matrix computation.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import torch

from edge_vision.data.dataset import CIFAR10_CLASSES


@dataclass
class EvaluationMetrics:
    """
    Structured metrics calculated over a complete evaluation dataset partition.
    """
    total_samples: int
    correct_predictions: int
    accuracy: float
    loss: float
    per_class_accuracy: Dict[str, float]
    per_class_counts: Dict[str, int]
    per_class_correct: Dict[str, int]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationTracker:
    """
    Accumulator for evaluation batch outputs, computing sample-weighted cross-entropy loss,
    overall top-1 accuracy, per-class accuracies, and full confusion matrix.
    """

    def __init__(self, class_names: Sequence[str] = CIFAR10_CLASSES) -> None:
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.reset()

    def reset(self) -> None:
        """Reset all counters and the confusion matrix."""
        self._running_weighted_loss: float = 0.0
        self._total_samples: int = 0
        self._correct_predictions: int = 0
        self._confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

    def update(
        self,
        batch_loss: float,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Updates the tracker with batch evaluation results.

        Parameters:
            batch_loss: Mean scalar loss for the batch.
            preds: Class predictions or raw logits tensor (shape: [B] or [B, C]).
            targets: Ground-truth target labels (shape: [B]).
            batch_size: Explicit batch size (inferred if None).
        """
        if batch_size is None:
            batch_size = targets.size(0)

        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)

        preds_cpu = preds.detach().cpu().long()
        targets_cpu = targets.detach().cpu().long()

        self._running_weighted_loss += float(batch_loss) * batch_size
        correct = int((preds_cpu == targets_cpu).sum().item())
        self._correct_predictions += correct
        self._total_samples += int(batch_size)

        # Update confusion matrix: row = true label, col = predicted label
        for t, p in zip(targets_cpu.view(-1), preds_cpu.view(-1)):
            t_idx = t.item()
            p_idx = p.item()
            if 0 <= t_idx < self.num_classes and 0 <= p_idx < self.num_classes:
                self._confusion_matrix[t_idx, p_idx] += 1

    def compute(self) -> EvaluationMetrics:
        """
        Computes aggregated evaluation metrics and validates mathematical consistency.

        Returns:
            EvaluationMetrics dataclass instance.
        """
        if self._total_samples == 0:
            empty_dict = {name: 0.0 for name in self.class_names}
            empty_counts = {name: 0 for name in self.class_names}
            return EvaluationMetrics(
                total_samples=0,
                correct_predictions=0,
                accuracy=0.0,
                loss=0.0,
                per_class_accuracy=empty_dict,
                per_class_counts=empty_counts,
                per_class_correct=empty_counts,
                confusion_matrix=[[0] * self.num_classes for _ in range(self.num_classes)],
            )

        mean_loss = self._running_weighted_loss / self._total_samples
        overall_accuracy = self._correct_predictions / self._total_samples

        per_class_accuracy: Dict[str, float] = {}
        per_class_counts: Dict[str, int] = {}
        per_class_correct: Dict[str, int] = {}

        sum_class_counts = 0
        sum_class_correct = 0

        for c_idx, name in enumerate(self.class_names):
            class_total = int(self._confusion_matrix[c_idx, :].sum().item())
            class_correct = int(self._confusion_matrix[c_idx, c_idx].item())
            class_acc = class_correct / class_total if class_total > 0 else 0.0

            per_class_accuracy[name] = round(class_acc, 6)
            per_class_counts[name] = class_total
            per_class_correct[name] = class_correct

            sum_class_counts += class_total
            sum_class_correct += class_correct

        # Verify mathematical integrity
        cm_total = int(self._confusion_matrix.sum().item())
        if cm_total != self._total_samples:
            raise ValueError(
                f"Confusion matrix sum ({cm_total}) does not equal total samples ({self._total_samples})"
            )
        if sum_class_counts != self._total_samples:
            raise ValueError(
                f"Sum of per-class sample counts ({sum_class_counts}) does not equal total samples ({self._total_samples})"
            )
        if sum_class_correct != self._correct_predictions:
            raise ValueError(
                f"Sum of diagonal confusion elements ({sum_class_correct}) does not equal correct predictions ({self._correct_predictions})"
            )

        cm_list: List[List[int]] = self._confusion_matrix.tolist()

        return EvaluationMetrics(
            total_samples=self._total_samples,
            correct_predictions=self._correct_predictions,
            accuracy=round(overall_accuracy, 6),
            loss=round(mean_loss, 6),
            per_class_accuracy=per_class_accuracy,
            per_class_counts=per_class_counts,
            per_class_correct=per_class_correct,
            confusion_matrix=cm_list,
        )
