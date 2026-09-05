"""Evaluation: curve metrics, bias/variance, paired statistics, kinetics, frame analysis."""

from .bias_variance import BiasVariance, decompose, decompose_by_condition, decompose_frame
from .curve_metrics import CurveMetrics, compute_metrics, metrics_frame
from .frames import failure_modes, frame_error_table, summarise_by_bin
from .kinetics import compare_kinetics, fit_two_tissue, patlak
from .stats import PairedComparison, adjust_pvalues, compare_conditions, paired_test

__all__ = [
    "CurveMetrics", "compute_metrics", "metrics_frame",
    "BiasVariance", "decompose", "decompose_frame", "decompose_by_condition",
    "PairedComparison", "paired_test", "compare_conditions", "adjust_pvalues",
    "patlak", "fit_two_tissue", "compare_kinetics",
    "frame_error_table", "summarise_by_bin", "failure_modes",
]
