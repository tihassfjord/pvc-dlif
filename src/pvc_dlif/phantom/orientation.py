"""
Orientation handling for NEMA NU-4 phantom images

This module provides functions to detect and correct image orientation
to ensure ROI detection works correctly regardless of how the image was acquired.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import zoom


def detect_phantom_orientation(image):
    """
    Detect the orientation of the NEMA NU-4 phantom based on intensity patterns.

    The phantom has:
    - Hot rods (high intensity)
    - Cold inserts (low intensity - water and air)
    - Uniform region (medium intensity)

    For NEMA NU-4, the phantom extends along its SHORTEST axis (typically ~128 slices)
    The transaxial plane is the largest (typically 184x184).

    Parameters
    ----------
    image : np.ndarray
        3D image array

    Returns
    -------
    dict
        Dictionary with orientation information:
        - 'axis': Which axis the phantom extends along (0, 1, or 2)
        - 'flip': Whether to flip along that axis
        - 'cold_end': Which end has cold regions ('start' or 'end')
    """
    # Check intensity distribution along each axis
    orientations = {}

    for axis in range(3):
        # Calculate mean intensity per slice along this axis
        slice_means = [np.mean(np.take(image, i, axis=axis))
                      for i in range(image.shape[axis])]

        # Calculate intensity variation
        variation = np.std(slice_means)

        # Find the non-zero region (where the phantom actually is)
        slice_means_array = np.array(slice_means)
        non_zero_indices = np.where(slice_means_array > np.max(slice_means_array) * 0.01)[0]
        
        if len(non_zero_indices) > 0:
            # Get the range where the phantom exists
            phantom_start = non_zero_indices[0]
            phantom_end = non_zero_indices[-1]
            phantom_length = phantom_end - phantom_start + 1
            
            # Look at first and last quarters OF THE PHANTOM REGION (not the whole volume)
            quarter_size = max(1, phantom_length // 4)
            first_quarter_idx = range(phantom_start, min(phantom_start + quarter_size, phantom_end + 1))
            last_quarter_idx = range(max(phantom_end - quarter_size + 1, phantom_start), phantom_end + 1)
            
            first_mean = np.mean([slice_means[i] for i in first_quarter_idx])
            last_mean = np.mean([slice_means[i] for i in last_quarter_idx])
        else:
            # No phantom detected on this axis
            first_mean = 0
            last_mean = 0
        
        # Determine if cold is at start or end (cold = lower intensity)
        cold_at_start = first_mean < last_mean

        orientations[axis] = {
            'variation': variation,
            'cold_at_start': cold_at_start,
            'slice_means': slice_means,
            'axis_length': image.shape[axis],
            'first_mean': first_mean,
            'last_mean': last_mean
        }

    # The phantom axis should be the SHORTEST axis with highest variation
    # (NEMA NU-4 is ~30mm diameter, extends ~50mm along length)
    # However, some acquisitions may have the phantom as the longest axis
    # So we check both and use the one with highest variation
    
    shortest_axis = min(orientations.keys(),
                       key=lambda k: orientations[k]['axis_length'])
    longest_axis = max(orientations.keys(),
                      key=lambda k: orientations[k]['axis_length'])
    highest_var_axis = max(orientations.keys(),
                          key=lambda k: orientations[k]['variation'])
    
    # If the longest axis has much higher variation, it's probably the phantom axis
    if orientations[longest_axis]['variation'] > orientations[shortest_axis]['variation'] * 2:
        best_axis = longest_axis
    else:
        best_axis = shortest_axis
    
    # Fallback to highest variation if needed
    if orientations[best_axis]['variation'] < 100:
        best_axis = highest_var_axis

    return {
        'axis': best_axis,
        'flip': not orientations[best_axis]['cold_at_start'],  # Cold should be at start
        'cold_end': 'start' if orientations[best_axis]['cold_at_start'] else 'end',
        'variation': orientations[best_axis]['variation'],
        'first_mean': orientations[best_axis]['first_mean'],
        'last_mean': orientations[best_axis]['last_mean']
    }


def reorient_for_nema_analysis(image_path, verbose=True):
    """
    Load and reorient image for NEMA NU-4 analysis.

    Ensures the image is oriented correctly so that:
    - Cold regions are at the beginning of the primary axis
    - Hot rods are at the end
    - Uniform region is in the middle

    Parameters
    ----------
    image_path : str or Path
        Path to NIfTI image file
    verbose : bool
        Print orientation information

    Returns
    -------
    tuple
        (reoriented_image, original_nifti, orientation_info)
    """
    # Load image
    nii = nib.load(str(image_path))
    image = nii.get_fdata()

    if verbose:
        print(f"Original image shape: {image.shape}")
        print(f"Original voxel size: {nii.header.get_zooms()[:3]}")

    # Handle 4D data (sum over time)
    if image.ndim == 4:
        if verbose:
            print(f"4D image detected, summing over time axis...")
        # Find time axis (usually smallest dimension or first/last)
        axis_sizes = list(image.shape)
        min_axis_size = min(axis_sizes)
        if min_axis_size < 100:
            time_axis = axis_sizes.index(min_axis_size)
        else:
            time_axis = 0 if image.shape[0] < image.shape[-1] else 3
        image = np.sum(image, axis=time_axis)
        if verbose:
            print(f"Summed over axis {time_axis}, new shape: {image.shape}")

    # Detect orientation
    orientation = detect_phantom_orientation(image)

    if verbose:
        print(f"\nDetected orientation:")
        print(f"  Primary axis: {orientation['axis']} (0=X, 1=Y, 2=Z)")
        print(f"  Need to flip: {orientation['flip']}")
        print(f"  Cold regions at: {orientation['cold_end']}")
        print(f"  Intensity variation: {orientation['variation']:.2f}")
        print(f"  First quarter mean: {orientation['first_mean']:.1f}")
        print(f"  Last quarter mean: {orientation['last_mean']:.1f}")
        print(f"  → Cold should be at START (lower intensity at beginning)")

    # Reorient if needed
    reoriented = image.copy()

    # The NEMA NU-4 phantom should have its long axis (through the phantom) as axis 2 (Z)
    # Target shape should be like (184, 184, 128) where Z is the phantom axis
    if orientation['axis'] != 2:
        if verbose:
            print(f"\nSwapping axes: moving axis {orientation['axis']} to position 2")
        # Move the detected phantom axis to position 2 (Z)
        if orientation['axis'] == 0:
            # Move axis 0 to position 2: (0,1,2) -> (1,2,0)
            reoriented = np.transpose(reoriented, (1, 2, 0))
        elif orientation['axis'] == 1:
            # Move axis 1 to position 2: (0,1,2) -> (0,2,1)
            reoriented = np.transpose(reoriented, (0, 2, 1))
        if verbose:
            print(f"New shape after swap: {reoriented.shape}")

    # Flip if cold regions are at wrong end
    if orientation['flip']:
        if verbose:
            print(f"\nFlipping along axis 2 (phantom axis) to put cold regions at start")
        reoriented = np.flip(reoriented, axis=2)

    # Verify the reorientation
    if verbose:
        print(f"\nVerifying reorientation:")
        print(f"  First 10 slices (Z) mean intensity: {np.mean(reoriented[:, :, :10]):.1f}")
        print(f"  Middle slices (Z) mean intensity: {np.mean(reoriented[:, :, reoriented.shape[2]//2-5:reoriented.shape[2]//2+5]):.1f}")
        print(f"  Last 10 slices (Z) mean intensity: {np.mean(reoriented[:, :, -10:]):.1f}")
        print(f"  (Cold → Uniform → Hot pattern expected along Z)")

    return reoriented, nii, orientation


def save_reoriented_image(reoriented_image, original_nii, output_path, verbose=True):
    """
    Save reoriented image as NIfTI with updated affine.
    
    Sets the orientation to RPS (Right-Posterior-Superior) which is the
    standard orientation for NEMA NU-4 phantom analysis.

    Parameters
    ----------
    reoriented_image : np.ndarray
        Reoriented 3D image array
    original_nii : nibabel.Nifti1Image
        Original NIfTI image object (for header information)
    output_path : str or Path
        Where to save the reoriented image
    verbose : bool
        Print save information
    """
    # Get voxel size from original
    voxel_size = original_nii.header.get_zooms()[:3]
    
    # Create RPS orientation affine matrix
    # RPS means: Right=-X, Posterior=-Y, Superior=+Z
    affine = np.array([
        [-voxel_size[0], 0, 0, 0],
        [0, -voxel_size[1], 0, 0],
        [0, 0, voxel_size[2], 0],
        [0, 0, 0, 1]
    ])

    new_nii = nib.Nifti1Image(reoriented_image, affine)

    # Copy relevant header information
    new_nii.header['descrip'] = b'Reoriented to RPS for NEMA NU-4 analysis'
    
    # Set sform to indicate alignment
    new_nii.header.set_sform(affine, code='aligned')
    new_nii.header.set_qform(affine, code='aligned')

    nib.save(new_nii, str(output_path))

    if verbose:
        print(f"\nSaved reoriented image to: {output_path}")
        print(f"  Orientation: RPS (Right-Posterior-Superior)")
        print(f"  Affine matrix:")
        print(f"    {affine}")


def quick_orientation_check(image_path):
    """
    Quick check of image orientation without full reorientation.

    Parameters
    ----------
    image_path : str or Path
        Path to NIfTI image

    Returns
    -------
    str
        Short summary of orientation
    """
    nii = nib.load(str(image_path))
    image = nii.get_fdata()

    if image.ndim == 4:
        image = np.sum(image, axis=0)

    orientation = detect_phantom_orientation(image)

    summary = f"Primary axis: {orientation['axis']}, "
    summary += f"Cold at {orientation['cold_end']}, "
    summary += f"Flip needed: {orientation['flip']}"

    return summary


def resample_to_target(image, current_voxel_size, target_voxel_size, verbose=True):
    """
    Resample image to target voxel size.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array
    current_voxel_size : tuple
        Current voxel dimensions (x, y, z) in mm
    target_voxel_size : tuple
        Target voxel dimensions (x, y, z) in mm
    verbose : bool
        Print resampling information
        
    Returns
    -------
    np.ndarray
        Resampled image
    """
    if verbose:
        print(f"\nResampling image...")
        print(f"  Current voxel size: {current_voxel_size}")
        print(f"  Target voxel size: {target_voxel_size}")
        print(f"  Current shape: {image.shape}")
    
    # Calculate zoom factors
    zoom_factors = [
        current_voxel_size[0] / target_voxel_size[0],
        current_voxel_size[1] / target_voxel_size[1],
        current_voxel_size[2] / target_voxel_size[2]
    ]
    
    if verbose:
        print(f"  Zoom factors: {zoom_factors}")
    
    # Resample using scipy zoom (trilinear interpolation)
    resampled = zoom(image, zoom_factors, order=1, mode='constant', cval=0.0)
    
    if verbose:
        print(f"  New shape: {resampled.shape}")
    
    return resampled
    summary += f"Flip needed: {orientation['flip']}"

    return summary


def estimate_3d_tilt(image, verbose=False):
    """
    Estimate the 3D tilt of the phantom by analyzing its principal axis.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array
    verbose : bool
        Print detailed information
    
    Returns
    -------
    dict
        Dictionary containing tilt angles and principal axis direction
    """
    # Threshold to get phantom region
    threshold = np.max(image) * 0.2
    binary = image > threshold
    
    # Get coordinates of phantom voxels
    coords = np.array(np.where(binary)).T  # Shape: (N, 3)
    
    if len(coords) == 0:
        return {'tilt_xy': 0.0, 'tilt_xz': 0.0, 'tilt_yz': 0.0}
    
    # Center the coordinates
    center = np.mean(coords, axis=0)
    coords_centered = coords - center
    
    # Compute covariance matrix
    cov_matrix = np.cov(coords_centered.T)
    
    # Find principal axes via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Principal axis (longest direction of phantom)
    principal_axis = eigenvectors[:, 0]
    
    # Ensure it points in positive Z direction
    if principal_axis[2] < 0:
        principal_axis = -principal_axis
    
    # Calculate tilt angles from Z-axis
    z_axis = np.array([0, 0, 1])
    
    # Angle between principal axis and Z-axis
    dot_product = np.dot(principal_axis, z_axis)
    angle_from_z = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    # More accurate calculation of rotation angles needed
    # For rotation around X-axis (fixes YZ plane tilt - seen in X slices)
    tilt_yz_rad = np.arctan2(principal_axis[1], principal_axis[2])
    tilt_yz = np.degrees(tilt_yz_rad)
    
    # For rotation around Y-axis (fixes XZ plane tilt - seen in Y slices)
    tilt_xz_rad = np.arctan2(principal_axis[0], principal_axis[2])
    tilt_xz = np.degrees(tilt_xz_rad)
    
    # For rotation around Z-axis (fixes XY plane rotation - seen in Z slices)
    tilt_xy_rad = np.arctan2(principal_axis[1], principal_axis[0])
    tilt_xy = np.degrees(tilt_xy_rad)
    
    if verbose:
        print(f"Principal axis: {principal_axis}")
        print(f"Eigenvalues: {eigenvalues}")
    
    return {
        'principal_axis': principal_axis,
        'angle_from_z': angle_from_z,
        'tilt_xy': tilt_xy,
        'tilt_xz': tilt_xz,
        'tilt_yz': tilt_yz
    }


def correct_3d_tilt(image, tilt_info, apply_all=True):
    """
    Correct 3D tilt by rotating the image to align phantom with Z-axis.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array
    tilt_info : dict
        Tilt information from estimate_3d_tilt
    apply_all : bool
        If True, apply all tilt corrections. If False, only XY rotation.
    
    Returns
    -------
    np.ndarray
        Tilt-corrected image
    """
    corrected = image.copy()
    
    # Get principal axis
    principal_axis = tilt_info['principal_axis']
    
    # Ensure principal axis points in positive Z direction
    if principal_axis[2] < 0:
        principal_axis = -principal_axis
    
    # Apply rotations to align phantom axis with Z-axis
    # The order matters: rotate around X first (YZ plane), then Y (XZ plane), then Z (XY plane)
    # NOTE: Using POSITIVE angles (not negative) to correct the tilt
    
    # 1. Rotate around X-axis to fix YZ plane tilt (this is what you see in X-slices)
    if abs(tilt_info['tilt_yz']) > 0.5:
        print(f"    Rotating {tilt_info['tilt_yz']:.2f}° around X-axis (YZ plane)")
        corrected = ndimage.rotate(corrected, tilt_info['tilt_yz'],
                                   axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
    
    # 2. Rotate around Y-axis to fix XZ plane tilt (seen in Y-slices)
    if abs(tilt_info['tilt_xz']) > 0.5:
        print(f"    Rotating {tilt_info['tilt_xz']:.2f}° around Y-axis (XZ plane)")
        corrected = ndimage.rotate(corrected, tilt_info['tilt_xz'],
                                   axes=(0, 2), reshape=False, order=1, mode='constant', cval=0)
    
    # 3. Rotate around Z-axis for XY alignment (seen in Z-slices)
    if abs(tilt_info['tilt_xy']) > 0.5:
        print(f"    Rotating {tilt_info['tilt_xy']:.2f}° around Z-axis (XY plane)")
        corrected = ndimage.rotate(corrected, tilt_info['tilt_xy'],
                                   axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
    
    return corrected


def estimate_rotation_angle(image, axis=2):
    """
    Estimate the rotation angle of the phantom in the plane perpendicular to the given axis.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array
    axis : int
        The axis perpendicular to the rotation plane (default: 2 for XY plane)
    
    Returns
    -------
    float
        Estimated rotation angle in degrees
    """
    # Get a slice through the middle of the phantom
    mid_slice_idx = image.shape[axis] // 2
    if axis == 0:
        slice_2d = image[mid_slice_idx, :, :]
    elif axis == 1:
        slice_2d = image[:, mid_slice_idx, :]
    else:  # axis == 2
        slice_2d = image[:, :, mid_slice_idx]
    
    # Threshold to get phantom region
    threshold = np.max(slice_2d) * 0.3
    binary = slice_2d > threshold
    
    # Find moments to calculate principal axis
    from scipy import ndimage
    
    # Calculate image moments
    y_coords, x_coords = np.nonzero(binary)
    if len(x_coords) == 0:
        return 0.0
    
    # Center of mass
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)
    
    # Second moments
    x_centered = x_coords - cx
    y_centered = y_coords - cy
    
    Ixx = np.sum(y_centered ** 2)
    Iyy = np.sum(x_centered ** 2)
    Ixy = -np.sum(x_centered * y_centered)
    
    # Principal axis angle
    angle = 0.5 * np.arctan2(2 * Ixy, Iyy - Ixx)
    angle_deg = np.degrees(angle)
    
    return angle_deg


def correct_rotation(image, angle_deg, axis=2, reshape=False):
    """
    Rotate image to correct for phantom rotation.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array
    angle_deg : float
        Rotation angle in degrees
    axis : int
        Axis perpendicular to rotation plane (0, 1, or 2)
    reshape : bool
        Whether to reshape the output array
    
    Returns
    -------
    np.ndarray
        Rotated image
    """
    # Determine rotation axes based on the perpendicular axis
    if axis == 0:
        axes = (1, 2)  # Rotate in YZ plane
    elif axis == 1:
        axes = (0, 2)  # Rotate in XZ plane
    else:  # axis == 2
        axes = (0, 1)  # Rotate in XY plane
    
    # Apply rotation
    rotated = ndimage.rotate(image, angle_deg, axes=axes, reshape=reshape, order=1)
    
    return rotated


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orientation.py <nifti_file> [output_file]")
        print("\nChecks orientation and optionally saves reoriented version")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("NEMA NU-4 Phantom Orientation Analysis")
    print("=" * 60)

    # Analyze orientation
    reoriented, nii, orientation = reorient_for_nema_analysis(input_file, verbose=True)

    # Save if output specified
    if output_file:
        save_reoriented_image(reoriented, nii, output_file, verbose=True)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
