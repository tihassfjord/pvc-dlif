"""Thesis output: figures and LaTeX tables built from the result tables."""

from .figures import save_figure
from .tables import comparison_table, metrics_summary_table, bias_variance_table, write_table

__all__ = [
    "save_figure", "comparison_table", "metrics_summary_table", "bias_variance_table", "write_table",
]
