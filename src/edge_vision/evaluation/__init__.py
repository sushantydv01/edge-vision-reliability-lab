"""
Standalone evaluation, metrics, confusion matrix, and reporting for Edge Vision Reliability Lab.
"""

from edge_vision.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationTracker,
)
from edge_vision.evaluation.evaluator import (
    convert_model_to_fp16,
    export_inference_checkpoint,
    resolve_evaluation_device,
    load_checkpoint_for_evaluation,
    get_evaluation_dataloader,
    evaluate_model,
    save_evaluation_report,
)
from edge_vision.evaluation.robustness import (
    REQUIRED_BENCHMARK_ARTIFACTS,
    ConditionModelResult,
    RobustnessAggregateSummary,
    RobustnessBenchmarkResult,
    RobustnessCondition,
    RobustnessEvaluator,
    check_existing_benchmark,
    compute_accuracy_degradation,
    compute_file_sha256,
    compute_model_robustness_summary,
    construct_condition_dataloader,
    get_standard_robustness_conditions,
    save_robustness_results,
    validate_output_directory,
)

__all__ = [
    "convert_model_to_fp16",
    "export_inference_checkpoint",
    "EvaluationMetrics",
    "EvaluationTracker",
    "resolve_evaluation_device",
    "load_checkpoint_for_evaluation",
    "get_evaluation_dataloader",
    "evaluate_model",
    "save_evaluation_report",
    "REQUIRED_BENCHMARK_ARTIFACTS",
    "ConditionModelResult",
    "RobustnessAggregateSummary",
    "RobustnessBenchmarkResult",
    "RobustnessCondition",
    "RobustnessEvaluator",
    "check_existing_benchmark",
    "compute_accuracy_degradation",
    "compute_file_sha256",
    "compute_model_robustness_summary",
    "construct_condition_dataloader",
    "get_standard_robustness_conditions",
    "save_robustness_results",
    "validate_output_directory",
]
