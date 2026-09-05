"""
Unified NEMA NU-4 Metrics
==========================

Compute NEMA NU-4 style performance metrics for small-animal PET phantom images.

This module provides a clean interface to the nema-nu4-benchmark code,
extracting recovery coefficients, spillover ratios, uniformity, and
other standard metrics.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import nibabel as nib

# Add nema-nu4-benchmark to path - search multiple possible locations
# Directory structure:
# A_UIT/
#   nema-nu4-benchmark/  ← Need to find this
#   MASTER/
#     tkinter_gui/
#       pvcgui/
#         pvc_core/      ← We are here
#           nema_metrics.py

_script_file = Path(__file__).resolve()
_pvc_core_dir = _script_file.parent
_pvcgui_dir = _pvc_core_dir.parent
_tkinter_gui_dir = _pvcgui_dir.parent
_master_dir = _tkinter_gui_dir.parent
_uit_dir = _master_dir.parent

_possible_paths = [
    _uit_dir / "nema-nu4-benchmark",  # ../../../nema-nu4-benchmark (relative to pvc_core)
    Path.cwd() / "nema-nu4-benchmark",  # ./nema-nu4-benchmark (current working directory)
    Path.home() / "nema-nu4-benchmark",  # ~/nema-nu4-benchmark
]

for _path in _possible_paths:
    if _path.exists():
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
        break


def _load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load or create default NEMA configuration.

    Parameters
    ----------
    config_path : Optional[str]
        Path to YAML config file. If None, use defaults for typical small-animal phantom.

    Returns
    -------
    Dict
        Configuration dictionary with 'image_quality' key containing phantom specs.
    """
    if config_path is not None:
        try:
            import yaml
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                # Return the image_quality section if it exists
                if isinstance(full_config, dict) and 'image_quality' in full_config:
                    return full_config
                # If the loaded config is directly the image_quality section, wrap it
                if isinstance(full_config, dict) and 'uniformity_roi' in full_config:
                    return {'image_quality': full_config}
                return full_config
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}: {e}. Using defaults.")

    # Default configuration for NEMA NU-4 small-animal phantom
    # Wrapped in 'image_quality' key for compatibility with analyze_image_quality
    return {
        'image_quality': {
            'uniformity_roi': {
                'diameter_mm': 22.5,
                'length_mm': 10.0,
            },
            'rod_diameters_mm': [1.0, 2.0, 3.0, 4.0, 5.0],
            'rod_offset_mm': 7.0,
            'uniform_region_length_mm': 13.0,  # Standard for 13mm cavity
            'spillover_roi': {
                'diameter_mm': 4.0,  # NEMA NU4-2008 Section 6.4.3
                'length_mm': 7.5,    # NEMA NU4-2008 Section 6.4.3
            },
            'voi_strategy': {
                'recovery_roi_scale': 2.0,  # ROI diameter = rod diameter × scale
            },
            'recovery_roi_scale': 2.0,  # Duplicate for direct access
            'calibration': {
                'reference_activity_bqml': None,  # Set if computing absolute RC
            },
        },
        'scanner': {
            'model': 'Unknown',
        },
        'reconstruction': {
            'algorithm': 'OSEM',
            'iterations': 10,
            'subsets': 1,
            'post_filter': 'None',
            'pvc_method': 'None',
        },
    }


def _get_image_metadata(image_nii: nib.Nifti1Image) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Extract image data and voxel size from NIfTI file.

    Parameters
    ----------
    image_nii : nib.Nifti1Image
        Loaded NIfTI image

    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float]]
        (image_data, voxel_size_mm)
    """
    data = np.asarray(image_nii.get_fdata(dtype=np.float32))

    # Extract voxel size from affine
    affine = image_nii.affine
    voxel_size = np.abs(np.diag(affine)[:3])
    voxel_size_mm = tuple(float(v) for v in voxel_size)

    return data, voxel_size_mm


def compute_nema_nu4_metrics(
    image_path: str,
    mask_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute NEMA NU-4 style metrics for a PET phantom image.

    This is the main entry point. It loads the image, applies the NEMA analysis,
    and returns a dictionary of metrics suitable for GUI display and CSV export.

    Parameters
    ----------
    image_path : str
        Path to PET image (NIfTI format)
    mask_path : Optional[str]
        Path to optional mask/ROI image. Default: None
    config_path : Optional[str]
        Path to YAML configuration file. Default: None (use built-in defaults)

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics with human-readable names:
        - "Uniformity_mean": Mean activity in uniform region
        - "Uniformity_std_pct": %STD in uniform region
        - "Uniformity_max": Maximum value
        - "Uniformity_min": Minimum value
        - "RC_Xmm": Recovery coefficient for X mm rod (e.g., "RC_1mm": 0.45)
        - "RC_std_pct_Xmm": %STD for X mm rod
        - "Spillover_Water": Spillover ratio for water insert
        - "Spillover_Air": Spillover ratio for air insert
        - "Spillover_std_pct_Water": %STD for water spillover
        - "Spillover_std_pct_Air": %STD for air spillover

    Raises
    ------
    FileNotFoundError
        If image file does not exist
    RuntimeError
        If NEMA analysis fails
    """
    # Validate input
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if mask_path is not None:
        mask_path_obj = Path(mask_path)
        if not mask_path_obj.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Load config
    config = _load_config(config_path)

    # Load image
    try:
        image_nii = nib.load(str(image_path_obj))
        image_data, voxel_size_mm = _get_image_metadata(image_nii)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

    # Load mask if provided
    mask_data = None
    if mask_path is not None:
        try:
            mask_nii = nib.load(str(mask_path_obj))
            mask_data = np.asarray(mask_nii.get_fdata(dtype=np.float32))
        except Exception as e:
            raise RuntimeError(f"Failed to load mask: {e}")

    # Run NEMA analysis
    try:
        from nema_nu4.image_quality_rc import analyze_image_quality
        from nema_nu4.types import ImageMeta
    except ImportError:
        raise RuntimeError(
            "Could not import nema_nu4 module. "
            "Ensure nema-nu4-benchmark is installed or in sys.path."
        )

    try:
        # Create metadata object
        meta = ImageMeta(
            voxel_size_mm=voxel_size_mm,
            isotope='F-18',
            half_life_sec=110.0 * 60,  # ~110 minutes
            energy_window_kev=(430, 650),  # Typical small-animal PET window
            coincidence_window_ns=3.6,  # Typical
            reconstruction={'algorithm': 'OSEM'},
            acquisition_duration_sec=None,
            activity_start_mbq=3.70939, #Mbq
            scanner_model=config.get('scanner', {}).get('model', 'Unknown'),
        )

        # Get image_quality config section
        image_quality_config = config.get('image_quality', config)

        # Compute metrics
        result = analyze_image_quality(image_data, meta, image_quality_config)

    except Exception as e:
        raise RuntimeError(f"NEMA analysis failed: {e}")

    # Convert result to flat dictionary with human-readable keys
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

    # Add recovery coefficients for each rod diameter
    for diameter_mm, rc_value in result.recovery_coefficients.items():
        key = f"RC_{diameter_mm}mm"
        output_dict[key] = float(rc_value)

    # Add RC standard deviations
    for diameter_mm, std_pct in result.recovery_std_pct.items():
        key = f"RC_std_pct_{diameter_mm}mm"
        output_dict[key] = float(std_pct)

    # Add absolute RC if available
    for diameter_mm, rc_abs in result.recovery_coefficients_absolute.items():
        key = f"RC_absolute_{diameter_mm}mm"
        output_dict[key] = float(rc_abs)

    return output_dict


def format_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics dictionary for nice display in GUI/reports.

    Parameters
    ----------
    metrics : Dict[str, float]
        Raw metrics from compute_nema_nu4_metrics

    Returns
    -------
    Dict[str, str]
        Formatted strings with appropriate precision
    """
    formatted = {}

    for key, value in metrics.items():
        if 'pct' in key.lower():
            # Percentage values: 2 decimal places
            formatted[key] = f"{value:.2f}%"
        elif 'RC_' in key:
            # Recovery coefficients: 3-4 decimal places
            formatted[key] = f"{value:.4f}"
        elif 'Spillover' in key and 'std' not in key:
            # Spillover ratios: 4 decimal places
            formatted[key] = f"{value:.4f}"
        else:
            # Others: 2 decimal places
            formatted[key] = f"{value:.2f}"

    return formatted


def compare_metrics(
    metrics_pre: Dict[str, float],
    metrics_post: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute percent change between pre and post metrics.

    Parameters
    ----------
    metrics_pre : Dict[str, float]
        Pre-PVC metrics
    metrics_post : Dict[str, float]
        Post-PVC metrics

    Returns
    -------
    Dict[str, float]
        Percent change for each metric (post - pre) / pre * 100
    """
    changes = {}

    for key in metrics_post.keys():
        if key in metrics_pre:
            pre_val = metrics_pre[key]
            post_val = metrics_post[key]

            if pre_val != 0:
                change_pct = (post_val - pre_val) / abs(pre_val) * 100
                changes[key] = change_pct
            else:
                changes[key] = 0.0 if post_val == 0 else np.inf

    return changes
