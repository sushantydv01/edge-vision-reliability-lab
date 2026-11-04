"""
Standalone evaluation, metrics, confusion matrix, and reporting for Edge Vision Reliability Lab.
"""

from edge_vision.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationTracker,
)
from edge_vision.evaluation.evaluator import (
    resolve_evaluation_device,
    load_checkpoint_for_evaluation,
    get_evaluation_dataloader,
    evaluate_model,
    save_evaluation_report,
)

__all__ = [
    "EvaluationMetrics",
    "EvaluationTracker",
    "resolve_evaluation_device",
    "load_checkpoint_for_evaluation",
    "get_evaluation_dataloader",
    "evaluate_model",
    "save_evaluation_report",
]
