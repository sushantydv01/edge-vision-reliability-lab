"""
Analysis package for Edge Vision Reliability Lab.
Provides cross-precision comparative evaluation and Pareto efficiency analysis.
"""

from edge_vision.analysis.cross_precision import (
    MODEL_METADATA,
    build_clean_accuracy_table,
    build_cpu_performance_table,
    build_model_size_table,
    build_pareto_analysis,
    build_per_corruption_table,
    build_per_severity_table,
    build_robustness_summary_table,
    load_cross_precision_data,
)
from edge_vision.analysis.pareto import compute_pareto_frontier, is_pareto_dominant

__all__ = [
    "MODEL_METADATA",
    "load_cross_precision_data",
    "build_model_size_table",
    "build_clean_accuracy_table",
    "build_robustness_summary_table",
    "build_per_corruption_table",
    "build_per_severity_table",
    "build_cpu_performance_table",
    "build_pareto_analysis",
    "compute_pareto_frontier",
    "is_pareto_dominant",
]

