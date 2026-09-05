"""
Automated NEMA NU-4 Benchmark Evaluation
=========================================

Complete pipeline that matches the NEMA_NU4_Benchmark.ipynb logic,
fully automated and callable from the GUI or scripts.

This module:
1. Loads a PET image
2. Computes all NEMA Section 6 metrics (uniformity, RC, spillover)
3. Generates comparison plots
4. Exports results to markdown/CSV
5. Returns metrics for GUI display
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import nibabel as nib
from datetime import datetime

# Add nema-nu4-benchmark to path
_script_file = Path(__file__).resolve()
_pvc_core_dir = _script_file.parent
_scripts_dir = _pvc_core_dir.parent
_simplified_project_dir = _scripts_dir.parent
_a_uit_dir = _simplified_project_dir.parent

_possible_paths = [
    _a_uit_dir / "nema-nu4-benchmark",  # ../../../nema-nu4-benchmark (most common)
    _simplified_project_dir / "nema-nu4-benchmark",  # ../../nema-nu4-benchmark
    Path.cwd() / "nema-nu4-benchmark",  # ./nema-nu4-benchmark (current dir)
]

for _path in _possible_paths:
    if _path.exists():
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
        break


def run_nema_benchmark(
    image_path: str,
    output_dir: Optional[str] = None,
    reference_image_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Run complete NEMA NU-4 benchmark evaluation on a PET image.

    This is the main entry point for automated NEMA evaluation.
    It replicates the NEMA_NU4_Benchmark.ipynb workflow.

    Parameters
    ----------
    image_path : str
        Path to PET image to evaluate (NIfTI format)
    output_dir : Optional[str]
        Directory for outputs (plots, reports, VOIs). Default: creates timestamped dir
    reference_image_path : Optional[str]
        Path to reference pre-PVC image for comparison. Default: None (no comparison)
    config_path : Optional[str]
        Path to YAML config file. Default: uses built-in NEMA NU-4 defaults

    Returns
    -------
    Dict
        Results dictionary containing:
        - 'metrics': dict of NEMA metrics (uniformity, RC, spillover, etc.)
        - 'image_quality_result': ImageQualityResult object
        - 'output_dir': Path where outputs were saved
        - 'plots': List of generated plot files
        - 'report_md': Markdown report text
    """

    try:
        from nema_nu4.image_quality_rc import analyze_image_quality
        from nema_nu4.types import ImageMeta
        from nema_nu4.plotting import plot_image_quality_comparison
    except ImportError:
        raise RuntimeError(
            "Could not import nema_nu4 module. "
            "Ensure nema-nu4-benchmark is installed or in sys.path."
        )

    # Validate input
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory with timestamp
    if output_dir is None:
        base_output_dir = Path.cwd() / "nema_outputs"
        base_output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"benchmark_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = _load_nema_config(config_path)

    # Load image
    try:
        image_nii = nib.load(str(image_path_obj))
        image_data = np.asarray(image_nii.get_fdata(dtype=np.float32))
        voxel_size = np.abs(np.diag(image_nii.affine)[:3])
        voxel_size_mm = tuple(float(v) for v in voxel_size)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Handle 4D data (sum over time)
    if image_data.ndim == 4:
        print(f"[INFO] Image is 4D {image_data.shape} - summing over time axis")
        # Detect time axis (smallest dimension typically)
        axis_sizes = list(image_data.shape)
        min_axis_size = min(axis_sizes)
        if min_axis_size < 100:
            time_axis = axis_sizes.index(min_axis_size)
        else:
            time_axis = 0
        image_data = np.sum(image_data, axis=time_axis)
        print(f"[INFO] After summation: {image_data.shape}")

    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image_data.shape}")

    # Create metadata
    meta = ImageMeta(
        voxel_size_mm=voxel_size_mm,
        isotope='F-18',
        half_life_sec=110.0 * 60,
        energy_window_kev=(430, 650),
        coincidence_window_ns=3.6,
        reconstruction={'algorithm': 'OSEM'},
        acquisition_duration_sec=None,
        activity_start_mbq=None,
        scanner_model=config.get('scanner', {}).get('model', 'Unknown'),
    )

    # Run NEMA analysis
    print(f"[INFO] Running NEMA NU-4 Image Quality analysis...")
    try:
        image_quality_config = config.get('image_quality', config)
        result = analyze_image_quality(image_data, meta, image_quality_config)
    except Exception as e:
        raise RuntimeError(f"NEMA analysis failed: {e}")

    # Convert to flat metrics dict for display
    metrics = _result_to_metrics_dict(result)

    # Generate plots
    plots_generated = []
    
    # Plot 1: Save slices
    try:
        slice_path = output_dir / 'phantom_slices.png'
        _plot_phantom_slices(image_data, slice_path)
        plots_generated.append(str(slice_path))
        print(f"[OK] Saved phantom slices: {slice_path.name}")
    except Exception as e:
        print(f"[WARN] Failed to generate slices plot: {e}")

    # Plot 2: Recovery coefficients bar chart
    try:
        rc_path = output_dir / 'recovery_coefficients.png'
        _plot_recovery_coefficients(result, rc_path)
        plots_generated.append(str(rc_path))
        print(f"[OK] Saved RC plot: {rc_path.name}")
    except Exception as e:
        print(f"[WARN] Failed to generate RC plot: {e}")

    # Plot 3: Metrics summary
    try:
        summary_path = output_dir / 'metrics_summary.png'
        _plot_metrics_summary(result, summary_path)
        plots_generated.append(str(summary_path))
        print(f"[OK] Saved metrics summary: {summary_path.name}")
    except Exception as e:
        print(f"[WARN] Failed to generate summary plot: {e}")

    # Generate report
    report_md = _generate_nema_report(
        result, image_path_obj, config, voxel_size_mm
    )
    report_path = output_dir / 'nema_report.md'
    with open(report_path, 'w') as f:
        f.write(report_md)
    print(f"[OK] Saved report: {report_path.name}")

    # Export metrics to CSV
    csv_path = output_dir / 'nema_metrics.csv'
    _export_metrics_csv(metrics, csv_path)
    print(f"[OK] Saved metrics CSV: {csv_path.name}")

    return {
        'metrics': metrics,
        'image_quality_result': result,
        'output_dir': str(output_dir),
        'plots': plots_generated,
        'report_md': report_md,
    }


def _load_nema_config(config_path: Optional[str] = None) -> Dict:
    """Load NEMA config or use built-in defaults."""
    if config_path is not None:
        try:
            import yaml
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                if isinstance(full_config, dict) and 'image_quality' in full_config:
                    return full_config
                if isinstance(full_config, dict) and 'uniformity_roi' in full_config:
                    return {'image_quality': full_config}
                return full_config
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}. Using defaults.")

    # Built-in defaults for NEMA NU-4 small-animal phantom
    return {
        'image_quality': {
            'uniformity_roi': {'diameter_mm': 22.5, 'length_mm': 10.0},
            'rod_diameters_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
            'rod_offset_mm': 7.0,
            'uniform_region_length_mm': 13.0,
            'spillover_roi': {'diameter_mm': 6.0, 'length_mm': 10.0},
            'voi_strategy': {'recovery_roi_scale': 2.0},
            'recovery_roi_scale': 2.0,
            'calibration': {'reference_activity_bqml': None},
        },
        'scanner': {'model': 'Unknown'},
        'reconstruction': {
            'algorithm': 'OSEM',
            'iterations': 10,
            'subsets': 1,
            'pvc_method': 'RL',
        },
    }


def _result_to_metrics_dict(result) -> Dict[str, float]:
    """Convert ImageQualityResult to flat metrics dictionary."""
    output_dict = {
        "Uniformity_mean": float(result.uniformity_mean),
        "Uniformity_std_pct": float(result.uniformity_std_pct),
        "Uniformity_max": float(result.uniformity_max),
        "Uniformity_min": float(result.uniformity_min),
        "Spillover_Water": float(result.spillover_ratio_water),
        "Spillover_Air": float(result.spillover_ratio_air),
        "Spillover_std_pct_Water": float(result.spillover_std_pct_water),
        "Spillover_std_pct_Air": float(result.spillover_std_pct_air),
    }

    for diameter_mm, rc_value in result.recovery_coefficients.items():
        key = f"RC_{diameter_mm}mm"
        output_dict[key] = float(rc_value)

    for diameter_mm, std_pct in result.recovery_std_pct.items():
        key = f"RC_std_pct_{diameter_mm}mm"
        output_dict[key] = float(std_pct)

    for diameter_mm, rc_abs in result.recovery_coefficients_absolute.items():
        key = f"RC_absolute_{diameter_mm}mm"
        output_dict[key] = float(rc_abs)

    return output_dict


def _plot_phantom_slices(image_data: np.ndarray, output_path: Path) -> None:
    """Generate 3-slice visualization of phantom."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    mid_z = image_data.shape[0] // 2
    mid_y = image_data.shape[1] // 2
    mid_x = image_data.shape[2] // 2
    
    axes[0].imshow(image_data[mid_z, :, :], cmap='hot')
    axes[0].set_title('Axial (XY)')
    axes[0].axis('off')
    
    axes[1].imshow(image_data[:, mid_y, :], cmap='hot', aspect='auto')
    axes[1].set_title('Coronal (XZ)')
    axes[1].axis('off')
    
    axes[2].imshow(image_data[:, :, mid_x], cmap='hot', aspect='auto')
    axes[2].set_title('Sagittal (YZ)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_recovery_coefficients(result, output_path: Path) -> None:
    """Generate RC bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    diameters = sorted(result.recovery_coefficients.keys())
    rc_values = [result.recovery_coefficients[d] for d in diameters]
    std_values = [result.recovery_std_pct[d] for d in diameters]
    
    x = np.arange(len(diameters))
    bars = ax.bar(x, rc_values, yerr=std_values, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Rod Diameter (mm)', fontweight='bold')
    ax.set_ylabel('Recovery Coefficient', fontweight='bold')
    ax.set_title('NEMA NU-4 Recovery Coefficients', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.1f}' for d in diameters])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rc_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics_summary(result, output_path: Path) -> None:
    """Generate metrics summary text/table visualization."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    summary_text = f"""
NEMA NU-4:2008 Image Quality Summary
{'='*50}

UNIFORMITY (NEMA Section 6.4.1)
  Mean Intensity:        {result.uniformity_mean:>10.2f}
  Std Dev (%):           {result.uniformity_std_pct:>10.2f}%
  Max Value:             {result.uniformity_max:>10.2f}
  Min Value:             {result.uniformity_min:>10.2f}

RECOVERY COEFFICIENTS (NEMA Section 6.4.2)
  Rod Diameter (mm)      Recovery Coeff    %STD
"""
    
    for d in sorted(result.recovery_coefficients.keys()):
        rc = result.recovery_coefficients[d]
        std = result.recovery_std_pct[d]
        summary_text += f"  {d:>15.1f}mm   {rc:>15.4f}  {std:>6.1f}%\n"
    
    summary_text += f"""
SPILLOVER RATIOS (NEMA Section 6.4.3)
  Water Insert:          {result.spillover_ratio_water:>10.4f} ± {result.spillover_std_pct_water:.1f}%
  Air Insert:            {result.spillover_ratio_air:>10.4f} ± {result.spillover_std_pct_air:.1f}%

"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_nema_report(result, image_path: Path, config: Dict, voxel_size_mm: Tuple) -> str:
    """Generate markdown report of NEMA evaluation."""
    
    report = f"""# NEMA NU-4:2008 Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Image**: {image_path.name}
**Voxel Size**: {voxel_size_mm[0]:.3f} × {voxel_size_mm[1]:.3f} × {voxel_size_mm[2]:.3f} mm³

## Overview

This report presents a complete NEMA NU-4:2008 image quality evaluation of the analyzed PET phantom image.

## 1. Uniformity (NEMA Section 6.4.1)

The uniformity region is a 22.5 mm diameter × 10 mm cylindrical VOI positioned in the uniform region of the phantom.

| Metric | Value |
|--------|-------|
| Mean Intensity | {result.uniformity_mean:.2f} |
| % Standard Deviation | {result.uniformity_std_pct:.2f}% |
| Maximum | {result.uniformity_max:.2f} |
| Minimum | {result.uniformity_min:.2f} |

**Interpretation**: %STD should be < 10% for good uniformity.

## 2. Recovery Coefficients (NEMA Section 6.4.2)

Recovery coefficients measure how well small structures are resolved. NEMA specifies ROI diameter = 2× physical rod diameter.

| Rod Diameter (mm) | Recovery Coefficient | %STD |
|-------------------|----------------------|------|
"""
    
    for d in sorted(result.recovery_coefficients.keys()):
        rc = result.recovery_coefficients[d]
        std = result.recovery_std_pct[d]
        report += f"| {d:.1f} | {rc:.4f} | {std:.2f}% |\n"
    
    report += f"""

**Interpretation**: 
- Larger rods typically have higher RC values
- RC typically increases from ~0.3 (1mm rods) to ~0.7-0.8 (5mm rods)
- PVC methods should increase RC values

## 3. Spillover Ratios (NEMA Section 6.4.3)

Spillover ratios measure background activity in cold regions. Lower is better.

| Region | Spillover Ratio | %STD |
|--------|-----------------|------|
| Water Insert | {result.spillover_ratio_water:.4f} | {result.spillover_std_pct_water:.2f}% |
| Air Insert | {result.spillover_ratio_air:.4f} | {result.spillover_std_pct_air:.2f}% |

**Interpretation**: Spillover should be < 0.05 for good background suppression.

## Conclusion

This evaluation provides objective, standardized metrics for assessing PET image quality according to the official NEMA NU-4:2008 standard.

---
*Report generated by Automated NEMA Benchmark Tool*
"""
    
    return report


def _export_metrics_csv(metrics: Dict[str, float], output_path: Path) -> None:
    """Export metrics to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, f"{value:.6f}"])
