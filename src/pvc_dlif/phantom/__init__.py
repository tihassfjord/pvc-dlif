"""
PVC Core Module
===============

Unified interface for PVC execution and NEMA metrics computation.

This module provides clean, reusable APIs that abstract away the complexity
of petpvc and NEMA metric calculation, making it easy to call from both GUI
and scripting contexts.
"""

from .pvc_runner import run_pvc
from .nema_metrics import compute_nema_nu4_metrics
from .benchmark import run_nema_benchmark
from .comparison import (
    export_comparison_csv,
    generate_comparison_summary,
    export_ranked_comparison,
    create_best_method_recommendation,
)
from .orientation import (
    detect_phantom_orientation,
    reorient_for_nema_analysis,
    save_reoriented_image,
    quick_orientation_check,
)
from .pls_pvc_matlab_port import (
    PLSPVCCorrector,
    pls_pvc_3d,
    create_mr_like_mask,
)

__all__ = [
    'run_pvc',
    'compute_nema_nu4_metrics',
    'run_nema_benchmark',
    'export_comparison_csv',
    'generate_comparison_summary',
    'export_ranked_comparison',
    'create_best_method_recommendation',
    'detect_phantom_orientation',
    'reorient_for_nema_analysis',
    'save_reoriented_image',
    'quick_orientation_check',
    'PLSPVCCorrector',
    'pls_pvc_3d',
    'create_mr_like_mask',
]
