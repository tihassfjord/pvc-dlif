"""
NEMA NU-4:2008 Section 3 - Spatial Resolution Measurement
=========================================================

This module implements spatial resolution measurements according to NEMA NU-4:2008 standard.
It analyzes point source images to extract FWHM (Full Width at Half Maximum) and
FWTM (Full Width at Tenth Maximum) in three orthogonal directions.

Requirements (NEMA NU-4:2008 Section 3):
- Point source: 22Na or 18F, < 0.3 mm in 10x10x10 mm acrylic cube
- Activity: low enough that dead time < 5%, randoms < 5%
- Measurement positions: axial center and 1/4 axial FOV
- Radial offsets: 5, 10, 15, 25 mm (+ 25 mm increments if field allows)
- Reconstruction: FBP (2D or 3D), no smoothing
- Pixel size: <= FWHM / 5
- Analysis: 1D profiles in radial, tangential, axial directions
- Peak finding: parabolic interpolation
- FWHM/FWTM: linear interpolation
- Report: mm, 2 decimal places

Usage:
    from .spatial_resolution import analyze_point_source, SpatialResolutionResult

    result = analyze_point_source(
        image_path="point_source.nii.gz",
        position_name="Center_5mm"
    )

    print(f"Radial FWHM: {result.fwhm_radial_mm:.2f} mm")
    print(f"Tangential FWHM: {result.fwhm_tangential_mm:.2f} mm")
    print(f"Axial FWHM: {result.fwhm_axial_mm:.2f} mm")
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass
import os

# Optional DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


@dataclass
class SpatialResolutionResult:
    """
    Results from spatial resolution analysis.

    Attributes
    ----------
    position_name : str
        Descriptive name of measurement position (e.g., "Center_5mm", "Quarter_10mm")
    peak_position_vox : Tuple[float, float, float]
        Peak position in voxel coordinates (i, j, k)
    peak_position_mm : Tuple[float, float, float]
        Peak position in physical coordinates (x, y, z) mm
    fwhm_radial_mm : float
        Full width at half maximum in radial direction (mm)
    fwhm_tangential_mm : float
        Full width at half maximum in tangential direction (mm)
    fwhm_axial_mm : float
        Full width at half maximum in axial direction (mm)
    fwtm_radial_mm : float
        Full width at tenth maximum in radial direction (mm)
    fwtm_tangential_mm : float
        Full width at tenth maximum in tangential direction (mm)
    fwtm_axial_mm : float
        Full width at tenth maximum in axial direction (mm)
    peak_value : float
        Maximum intensity value at peak
    """
    position_name: str
    peak_position_vox: Tuple[float, float, float]
    peak_position_mm: Tuple[float, float, float]
    fwhm_radial_mm: float
    fwhm_tangential_mm: float
    fwhm_axial_mm: float
    fwtm_radial_mm: float
    fwtm_tangential_mm: float
    fwtm_axial_mm: float
    peak_value: float
    radial_profile: Optional[np.ndarray] = None
    tangential_profile: Optional[np.ndarray] = None
    axial_profile: Optional[np.ndarray] = None


def read_dicom_series(path: str, prefix: str = "") -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Read DICOM series from directory (adapted from Daniel Sakarli's code).

    Handles various DICOM formats including dynamic PET data.

    Parameters
    ----------
    path : str
        Directory containing DICOM files
    prefix : str, optional
        Filter files by prefix

    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float]]
        (image_data, voxel_size_mm) where image_data is (Y, X, Z) format

    Raises
    ------
    RuntimeError
        If no valid DICOM files found or pydicom not available
    """
    if not DICOM_AVAILABLE:
        raise RuntimeError(
            "pydicom not installed. Install with: pip install pydicom\n"
            "Alternatively, convert DICOM to NIfTI first."
        )

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path must be a directory: {path}")

    # Get list of files
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if prefix:
        files = [f for f in files if f.startswith(prefix)]

    if not files:
        raise RuntimeError(f'No DICOM files found in {path}')

    # Read DICOM files
    meta, slices = [], []
    for f in sorted(files):
        try:
            ds = pydicom.dcmread(os.path.join(path, f))
            meta.append(ds)
            slices.append(ds.pixel_array.astype(np.float64))
        except Exception:
            pass  # Skip non-DICOM files

    if not slices:
        raise RuntimeError(f'No valid DICOM files in {path}')

    # Handle different cases
    if len(slices) == 1:
        arr = slices[0]

        # If 3D but actually 4D dynamic PET data stored as (total_frames, Y, X)
        if arr.ndim == 3 and arr.shape[0] > arr.shape[1]:  # More frames than spatial dims
            # Likely dynamic PET: (time*Z, Y, X) needs to be reshaped to (time, Z, Y, X)
            total_frames, ny, nx = arr.shape

            # Try common time frame counts
            for n_time in [42, 30, 20, 15, 10]:
                if total_frames % n_time == 0:
                    nz = total_frames // n_time
                    print(f"  ℹ️ Reshaping 3D→4D: {arr.shape} → ({n_time} time, {nz} Z, {ny} Y, {nx} X)")
                    arr = arr.reshape(n_time, nz, ny, nx)
                    # Sum over time
                    arr = arr.sum(axis=0)  # (time, Z, Y, X) → (Z, Y, X)
                    # Transpose to standard (Y, X, Z) format
                    arr = arr.transpose(1, 2, 0)  # (Z, Y, X) → (Y, X, Z)
                    print(f"Final shape: {arr.shape}")
                    break

        # If truly 4D, sum over time
        elif arr.ndim == 4:
            print(f"4D data detected: {arr.shape}, summing time...")
            arr = arr.sum(axis=0)  # (time, Y, X, Z) → (Y, X, Z)

        # If 2D, add Z dimension
        elif arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        image_data = arr
    else:
        # Multiple 2D files - stack along Z axis
        image_data = np.stack(slices, axis=2)

    # Extract voxel size from DICOM metadata
    ds_ref = meta[0] if len(meta) == 1 else meta[len(meta) // 2]

    try:
        dx, dy = map(float, ds_ref.PixelSpacing)
        dz = float(ds_ref.SliceThickness)
        voxel_size = (dx, dy, dz)
    except (AttributeError, KeyError):
        print("Warning: Could not extract voxel size from DICOM. Using default 1.0 mm.")
        voxel_size = (1.0, 1.0, 1.0)

    return image_data, voxel_size


def _find_peak_parabolic(profile: np.ndarray, initial_peak_idx: int) -> float:
    """
    Find peak position using parabolic interpolation (NEMA requirement).

    Parameters
    ----------
    profile : np.ndarray
        1D intensity profile
    initial_peak_idx : int
        Index of maximum value in profile

    Returns
    -------
    float
        Refined peak position (fractional index)
    """
    if initial_peak_idx == 0 or initial_peak_idx == len(profile) - 1:
        return float(initial_peak_idx)

    # Use 3 points around peak for parabolic fit
    y0 = profile[initial_peak_idx - 1]
    y1 = profile[initial_peak_idx]
    y2 = profile[initial_peak_idx + 1]

    # Parabolic interpolation formula
    # Peak offset from center point
    denominator = 2 * (2*y1 - y0 - y2)
    if abs(denominator) < 1e-10:
        return float(initial_peak_idx)

    offset = (y2 - y0) / denominator

    # Clamp offset to reasonable range
    offset = np.clip(offset, -1.0, 1.0)

    return initial_peak_idx + offset


def _find_width_at_fraction(
    profile: np.ndarray,
    peak_idx: float,
    peak_value: float,
    fraction: float = 0.5
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find width at specified fraction of peak using linear interpolation (NEMA requirement).

    Parameters
    ----------
    profile : np.ndarray
        1D intensity profile
    peak_idx : float
        Peak position (fractional index)
    peak_value : float
        Peak intensity value
    fraction : float
        Fraction of peak (0.5 for FWHM, 0.1 for FWTM)

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (left_crossing, right_crossing) in fractional indices, or (None, None) if not found
    """
    threshold = peak_value * fraction
    peak_idx_int = int(np.round(peak_idx))

    # Find left crossing (going left from peak)
    left_crossing = None
    for i in range(peak_idx_int, 0, -1):
        if profile[i] >= threshold > profile[i-1]:
            # Linear interpolation
            slope = (profile[i] - profile[i-1])
            if abs(slope) > 1e-10:
                frac = (threshold - profile[i-1]) / slope
                left_crossing = (i - 1) + frac
            else:
                left_crossing = float(i)
            break

    # Find right crossing (going right from peak)
    right_crossing = None
    for i in range(peak_idx_int, len(profile) - 1):
        if profile[i] >= threshold > profile[i+1]:
            # Linear interpolation
            slope = (profile[i+1] - profile[i])
            if abs(slope) > 1e-10:
                frac = (threshold - profile[i]) / slope
                right_crossing = i + frac
            else:
                right_crossing = float(i)
            break

    return left_crossing, right_crossing


def _extract_1d_profile(
    data: np.ndarray,
    center: Tuple[int, int, int],
    direction: str,
    length: int = 50
) -> np.ndarray:
    """
    Extract 1D profile from 3D data along specified direction.

    Parameters
    ----------
    data : np.ndarray
        3D image data
    center : Tuple[int, int, int]
        Center position (i, j, k) in voxel coordinates
    direction : str
        Direction: 'radial' (i), 'tangential' (j), or 'axial' (k)
    length : int
        Profile length in voxels (centered on peak)

    Returns
    -------
    np.ndarray
        1D intensity profile
    """
    ci, cj, ck = center
    half_len = length // 2

    if direction == 'radial':
        # Profile along i direction (first axis)
        i_start = max(0, ci - half_len)
        i_end = min(data.shape[0], ci + half_len)
        profile = data[i_start:i_end, cj, ck]
    elif direction == 'tangential':
        # Profile along j direction (second axis)
        j_start = max(0, cj - half_len)
        j_end = min(data.shape[1], cj + half_len)
        profile = data[ci, j_start:j_end, ck]
    elif direction == 'axial':
        # Profile along k direction (third axis)
        k_start = max(0, ck - half_len)
        k_end = min(data.shape[2], ck + half_len)
        profile = data[ci, cj, k_start:k_end]
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'radial', 'tangential', or 'axial'.")

    return profile


def analyze_point_source(
    image_path: str,
    position_name: str = "Unknown",
    search_radius: int = 10,
    profile_length: int = 50,
) -> SpatialResolutionResult:
    """
    Analyze point source image to extract spatial resolution metrics.

    This function implements NEMA NU-4:2008 Section 3 requirements for spatial resolution
    measurement. It finds the point source peak, extracts 1D profiles in three orthogonal
    directions, and computes FWHM and FWTM using the specified interpolation methods.

    Supports both NIfTI files and DICOM directories.

    Parameters
    ----------
    image_path : str
        Path to NIfTI image (.nii, .nii.gz) OR directory containing DICOM series
    position_name : str
        Descriptive name for this measurement position (e.g., "Center_5mm")
    search_radius : int
        Radius around initial peak to refine peak position (voxels)
    profile_length : int
        Length of 1D profiles to extract (voxels)

    Returns
    -------
    SpatialResolutionResult
        Comprehensive resolution metrics

    Raises
    ------
    FileNotFoundError
        If image file/directory does not exist
    ValueError
        If point source cannot be detected

    Notes
    -----
    NEMA NU-4:2008 Requirements:
    - Peak finding via parabolic interpolation
    - FWHM/FWTM via linear interpolation
    - Report values in mm with 2 decimal places

    Input Formats:
    - NIfTI: .nii or .nii.gz file
    - DICOM: Directory containing DICOM series
    """
    # Detect input type and load image
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Path not found: {image_path}")

    # Check if it's a directory (DICOM) or file (NIfTI)
    if image_path_obj.is_dir():
        # DICOM directory
        print(f"Loading DICOM series from: {image_path}")
        data, voxel_size = read_dicom_series(str(image_path_obj))
        # Ensure data is in (i, j, k) format for consistency
        # DICOM reader returns (Y, X, Z), we need to transpose to (i, j, k)
        # For point source, let's keep it as is since we analyze all 3 directions anyway
        data = data.astype(np.float32)
    else:
        # NIfTI file
        print(f"Loading NIfTI file: {image_path}")
        nii = nib.load(str(image_path_obj))
        data = np.asarray(nii.get_fdata(dtype=np.float32))

        # Extract voxel size from affine
        affine = nii.affine
        voxel_size = np.abs(np.diag(affine)[:3])

    # Find initial peak position (coarse)
    peak_idx_coarse = np.unravel_index(np.argmax(data), data.shape)
    peak_value = float(data[peak_idx_coarse])

    if peak_value <= 0:
        raise ValueError("No positive peak detected in image")

    # Refine peak position using center of mass in local region
    i, j, k = peak_idx_coarse
    i_min = max(0, i - search_radius)
    i_max = min(data.shape[0], i + search_radius)
    j_min = max(0, j - search_radius)
    j_max = min(data.shape[1], j + search_radius)
    k_min = max(0, k - search_radius)
    k_max = min(data.shape[2], k + search_radius)

    local_region = data[i_min:i_max, j_min:j_max, k_min:k_max]
    com = center_of_mass(local_region)

    peak_i = i_min + com[0]
    peak_j = j_min + com[1]
    peak_k = k_min + com[2]

    peak_vox = (peak_i, peak_j, peak_k)
    peak_mm = tuple(peak_vox[i] * voxel_size[i] for i in range(3))

    # Extract 1D profiles
    center_int = (int(np.round(peak_i)), int(np.round(peak_j)), int(np.round(peak_k)))

    radial_profile = _extract_1d_profile(data, center_int, 'radial', profile_length)
    tangential_profile = _extract_1d_profile(data, center_int, 'tangential', profile_length)
    axial_profile = _extract_1d_profile(data, center_int, 'axial', profile_length)

    # Analyze each profile
    def analyze_profile(profile: np.ndarray, voxel_size_mm: float) -> Tuple[float, float]:
        """Analyze single profile to get FWHM and FWTM."""
        peak_idx_initial = np.argmax(profile)
        peak_val = profile[peak_idx_initial]

        # Parabolic interpolation for peak
        peak_idx_refined = _find_peak_parabolic(profile, peak_idx_initial)

        # FWHM (50% of peak)
        left_hm, right_hm = _find_width_at_fraction(profile, peak_idx_refined, peak_val, 0.5)
        if left_hm is not None and right_hm is not None:
            fwhm_vox = right_hm - left_hm
            fwhm_mm = fwhm_vox * voxel_size_mm
        else:
            fwhm_mm = np.nan

        # FWTM (10% of peak)
        left_tm, right_tm = _find_width_at_fraction(profile, peak_idx_refined, peak_val, 0.1)
        if left_tm is not None and right_tm is not None:
            fwtm_vox = right_tm - left_tm
            fwtm_mm = fwtm_vox * voxel_size_mm
        else:
            fwtm_mm = np.nan

        return fwhm_mm, fwtm_mm

    fwhm_radial, fwtm_radial = analyze_profile(radial_profile, voxel_size[0])
    fwhm_tangential, fwtm_tangential = analyze_profile(tangential_profile, voxel_size[1])
    fwhm_axial, fwtm_axial = analyze_profile(axial_profile, voxel_size[2])

    result = SpatialResolutionResult(
        position_name=position_name,
        peak_position_vox=peak_vox,
        peak_position_mm=peak_mm,
        fwhm_radial_mm=fwhm_radial,
        fwhm_tangential_mm=fwhm_tangential,
        fwhm_axial_mm=fwhm_axial,
        fwtm_radial_mm=fwtm_radial,
        fwtm_tangential_mm=fwtm_tangential,
        fwtm_axial_mm=fwtm_axial,
        peak_value=peak_value,
        radial_profile=radial_profile,
        tangential_profile=tangential_profile,
        axial_profile=axial_profile,
    )

    return result


def plot_resolution_profiles(
    result: SpatialResolutionResult,
    output_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create visualization of 1D profiles with FWHM/FWTM markers.

    Parameters
    ----------
    result : SpatialResolutionResult
        Resolution analysis results
    output_path : Optional[str]
        Path to save figure (if None, no save)
    show : bool
        Whether to display figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    profiles = [
        (result.radial_profile, 'Radial', result.fwhm_radial_mm, result.fwtm_radial_mm),
        (result.tangential_profile, 'Tangential', result.fwhm_tangential_mm, result.fwtm_tangential_mm),
        (result.axial_profile, 'Axial', result.fwhm_axial_mm, result.fwtm_axial_mm),
    ]

    for ax, (profile, direction, fwhm, fwtm) in zip(axes, profiles):
        if profile is None:
            continue

        x = np.arange(len(profile))
        ax.plot(x, profile, 'b-', linewidth=2, label='Profile')

        peak_idx = np.argmax(profile)
        peak_val = profile[peak_idx]

        # Mark FWHM
        ax.axhline(peak_val * 0.5, color='r', linestyle='--', alpha=0.7, label=f'FWHM={fwhm:.2f} mm')

        # Mark FWTM
        ax.axhline(peak_val * 0.1, color='g', linestyle='--', alpha=0.7, label=f'FWTM={fwtm:.2f} mm')

        ax.set_xlabel('Position (voxels)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{direction} Direction')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Spatial Resolution: {result.position_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def generate_resolution_report(
    results: List[SpatialResolutionResult],
    output_path: str
) -> None:
    """
    Generate NEMA-compliant resolution report (CSV format).

    Parameters
    ----------
    results : List[SpatialResolutionResult]
        List of resolution measurements
    output_path : str
        Path to output CSV file
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Position',
            'Peak_X_mm', 'Peak_Y_mm', 'Peak_Z_mm',
            'FWHM_Radial_mm', 'FWHM_Tangential_mm', 'FWHM_Axial_mm',
            'FWTM_Radial_mm', 'FWTM_Tangential_mm', 'FWTM_Axial_mm',
            'Peak_Value'
        ])

        # Data rows (NEMA requires 2 decimal places for mm)
        for result in results:
            writer.writerow([
                result.position_name,
                f"{result.peak_position_mm[0]:.2f}",
                f"{result.peak_position_mm[1]:.2f}",
                f"{result.peak_position_mm[2]:.2f}",
                f"{result.fwhm_radial_mm:.2f}",
                f"{result.fwhm_tangential_mm:.2f}",
                f"{result.fwhm_axial_mm:.2f}",
                f"{result.fwtm_radial_mm:.2f}",
                f"{result.fwtm_tangential_mm:.2f}",
                f"{result.fwtm_axial_mm:.2f}",
                f"{result.peak_value:.4f}"
            ])


def batch_analyze_resolution(
    image_paths: List[str],
    position_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> List[SpatialResolutionResult]:
    """
    Batch process multiple point source images.

    Parameters
    ----------
    image_paths : List[str]
        List of paths to point source images
    position_names : Optional[List[str]]
        Names for each position (auto-generated if None)
    output_dir : Optional[str]
        Directory to save results (no save if None)

    Returns
    -------
    List[SpatialResolutionResult]
        Results for all positions
    """
    if position_names is None:
        position_names = [f"Position_{i+1}" for i in range(len(image_paths))]

    if len(position_names) != len(image_paths):
        raise ValueError("Number of position names must match number of images")

    results = []

    for img_path, pos_name in zip(image_paths, position_names):
        print(f"Analyzing: {pos_name}...")
        try:
            result = analyze_point_source(img_path, pos_name)
            results.append(result)

            print(f"  Radial FWHM: {result.fwhm_radial_mm:.2f} mm")
            print(f"  Tangential FWHM: {result.fwhm_tangential_mm:.2f} mm")
            print(f"  Axial FWHM: {result.fwhm_axial_mm:.2f} mm")

            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Save individual plot
                plot_path = output_dir_path / f"{pos_name}_profiles.png"
                plot_resolution_profiles(result, str(plot_path), show=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save combined report
    if output_dir and results:
        output_dir_path = Path(output_dir)
        report_path = output_dir_path / "spatial_resolution_report.csv"
        generate_resolution_report(results, str(report_path))
        print(f"\nReport saved: {report_path}")

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spatial_resolution.py <point_source_image.nii.gz> [position_name]")
        sys.exit(1)

    image_path = sys.argv[1]
    position_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown"

    result = analyze_point_source(image_path, position_name)

    print("\n" + "="*60)
    print(f"NEMA NU-4:2008 Spatial Resolution Results")
    print(f"Position: {result.position_name}")
    print("="*60)
    print(f"\nPeak Position (mm): ({result.peak_position_mm[0]:.2f}, "
          f"{result.peak_position_mm[1]:.2f}, {result.peak_position_mm[2]:.2f})")
    print(f"\nFWHM (mm):")
    print(f"  Radial:      {result.fwhm_radial_mm:.2f}")
    print(f"  Tangential:  {result.fwhm_tangential_mm:.2f}")
    print(f"  Axial:       {result.fwhm_axial_mm:.2f}")
    print(f"\nFWTM (mm):")
    print(f"  Radial:      {result.fwtm_radial_mm:.2f}")
    print(f"  Tangential:  {result.fwtm_tangential_mm:.2f}")
    print(f"  Axial:       {result.fwtm_axial_mm:.2f}")
    print("="*60)

    plot_resolution_profiles(result)
