"""
Unified PVC Runner
==================

Clean interface to run partial volume correction using petpvc.

This module wraps the PETPVCWrapper from unified_pvc to provide a
consistent interface for GUI and scripting use.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import nibabel as nib

# Add unified_pvc to path - search multiple possible locations
# Directory structure:
# A_UIT/
#   MASTER/
#     unified_pvc/        ← Need to find this
#     tkinter_gui/
#       pvcgui/
#         pvc_core/       ← We are here
#           pvc_runner.py

_script_file = Path(__file__).resolve()
_pvc_core_dir = _script_file.parent
_pvcgui_dir = _pvc_core_dir.parent
_tkinter_gui_dir = _pvcgui_dir.parent
_master_dir = _tkinter_gui_dir.parent

_possible_paths = [
    _pvcgui_dir / "unified_pvc",  # ../../unified_pvc (local copy in scripts/)
    _master_dir / "MASTER" / "unified_pvc",  # ../../../MASTER/unified_pvc (original location)
    _master_dir / "unified_pvc",  # ../../../unified_pvc (alternative location)
    Path.cwd() / "unified_pvc",  # ./unified_pvc (current working directory)
    Path.home() / "unified_pvc",  # ~/unified_pvc
]

for _path in _possible_paths:
    if _path.exists():
        if str(_path.parent) not in sys.path:
            sys.path.insert(0, str(_path.parent))
        break


def run_pvc(
    pet_image_path: str,
    output_path: str,
    method: str = "RL",
    fwhm_xyz: Tuple[float, float, float] = (0.849, 0.798, 1.023),
    iterations: int = 10,
    mask_path: Optional[str] = None,
    deconvolution_iterations: Optional[int] = None,
    alpha: Optional[float] = None,
    stopping_criterion: Optional[float] = None,
    disable_non_negativity: bool = False,
    debug: bool = False,
    pls_mu: float = 17.0,
    pls_use_mr_like_mask: bool = False,
    pls_threshold_percentile: float = 30.0,
     pls_smooth_sigma=2.0
) -> str:
    """
    Execute partial volume correction on a PET image.

    Parameters
    ----------
    pet_image_path : str
        Path to input PET image (NIfTI format, .nii or .nii.gz)
    output_path : str
        Path where PVC-corrected image will be saved
    method : str
        PVC method: "RL", "VC", "PLS", "IY", "MG", "GTM", "RBV", "MTC", etc.
        Default: "RL" (Richardson-Lucy)
    fwhm_xyz : Tuple[float, float, float]
        PSF FWHM in (x, y, z) mm. Default: measured for typical small-animal scanner
    iterations : int
        Number of iterations. Behavior depends on method:
        - For IY (iterative Yang): uses -n flag
        - For RL, VC, etc: uses -k flag (deconvolution iterations)
        Default: 10
    mask_path : Optional[str]
        Path to optional mask image (VOI or anatomical mask). Default: None (auto-mask generated)
    deconvolution_iterations : Optional[int]
        DEPRECATED: Use `iterations` parameter instead.
        Number of deconvolution iterations (-k flag). Used by RL and others.
        If provided, overrides `iterations` for RL/VC methods.
        Default: None
    alpha : Optional[float]
        Alpha parameter for deconvolution (-a flag). Default: None (uses petpvc default, typically 1.5)
    stopping_criterion : Optional[float]
        Stopping criterion for convergence (-s flag). Default: None (uses petpvc default, typically 0.01)
    disable_non_negativity : bool
        If True, turns off non-negativity constraint (-0 flag). Default: False (constraint enabled)
    debug : bool
        If True, prints debug information (-d flag). Default: False
    pls_mu : float
        Data fidelity parameter for PLS method (MATLAB: μ). Default: 17.0
        Only used when method="PLS"
    pls_use_mr_like_mask : bool
        If True and mask_path is None, creates an MR-like anatomical mask from PET
        using thresholding. Can improve edge preservation. Default: False
        Only used when method="PLS"
    pls_threshold_percentile : float
        Percentile threshold for MR-like mask generation (20-50 range). Default: 30.0
        Only used when method="PLS" and pls_use_mr_like_mask=True

    Returns
    -------
    str
        Path to the output PVC-corrected image

    Notes
    -----
    **Method-specific iteration handling:**
    
    - **IY (Iterative Yang)**: Uses `-n` flag (number of Yang iterations)
    - **RL, VC, GTM, RBV, MG, MTC, STC, LABBE**: Use `-k` flag (deconvolution iterations)
    
    The `iterations` parameter automatically maps to the correct flag based on method.

    Raises
    ------
    FileNotFoundError
        If input image file does not exist
    ValueError
        If method is unsupported or parameters are invalid
    RuntimeError
        If petpvc execution fails

    Examples
    --------
    >>> # Standard RL with 10 deconvolution iterations
    >>> pvc_path = run_pvc(
    ...     pet_image_path="/data/pet_scan.nii.gz",
    ...     output_path="/data/pet_scan_RL.nii.gz",
    ...     method="RL",
    ...     iterations=10
    ... )
    
    >>> # IY with 15 Yang iterations
    >>> pvc_path = run_pvc(
    ...     pet_image_path="/data/pet_scan.nii.gz",
    ...     output_path="/data/pet_scan_IY.nii.gz",
    ...     method="IY",
    ...     iterations=15
    ... )
    
    >>> # RL with custom alpha
    >>> pvc_path = run_pvc(
    ...     pet_image_path="/data/pet_scan.nii.gz",
    ...     output_path="/data/pet_scan_RL.nii.gz",
    ...     method="RL",
    ...     iterations=10,
    ...     alpha=1.8
    ... )

    >>> # PLS with custom mu and MR-like mask generation
    >>> pvc_path = run_pvc(
    ...     pet_image_path="/data/pet_scan.nii.gz",
    ...     output_path="/data/pet_scan_PLS.nii.gz",
    ...     method="PLS",
    ...     iterations=100,
    ...     pls_mu=17.0,
    ...     pls_use_mr_like_mask=True,
    ...     pls_threshold_percentile=30.0
    ... )
    """
    # Validate inputs
    pet_path = Path(pet_image_path)
    if not pet_path.exists():
        raise FileNotFoundError(f"PET image not found: {pet_image_path}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mask_path is not None:
        mask_path_obj = Path(mask_path)
        if not mask_path_obj.exists():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")

    # Validate method
    valid_methods = {
        "RL", "VC", "GTM", "RBV", "IY", "MG", "MTC", "STC", "LABBE",
        "RBV+VC", "RBV+RL", "LABBE+RBV", "LABBE+RBV+VC", "LABBE+RBV+RL",
        "IY+VC", "IY+RL",
        "MG+VC", "MG+RL",
        "MTC+VC", "MTC+RL", "LABBE+MTC", "LABBE+MTC+VC", "LABBE+MTC+RL",
        "PLS",  # MATLAB port with GPU acceleration
    }
    if method not in valid_methods:
        raise ValueError(
            f"Unsupported PVC method: {method}. "
            f"Supported methods: {', '.join(sorted(valid_methods))}"
        )

    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")

    if not all(f > 0 for f in fwhm_xyz):
        raise ValueError(f"FWHM values must be positive, got {fwhm_xyz}")

    # Validate optional parameters
    if deconvolution_iterations is not None and deconvolution_iterations <= 0:
        raise ValueError(f"Deconvolution iterations must be positive, got {deconvolution_iterations}")

    if alpha is not None and alpha <= 0:
        raise ValueError(f"Alpha must be positive, got {alpha}")

    if stopping_criterion is not None and stopping_criterion <= 0:
        raise ValueError(f"Stopping criterion must be positive, got {stopping_criterion}")

    # Handle PLS method separately (uses MATLAB port, not PETPVC)
    if method == "PLS":
        from .pls_pvc_matlab_port import pls_pvc_3d

        # Load input image
        try:
            pet_nii = nib.load(str(pet_path))
            pet_data = np.asarray(pet_nii.get_fdata(dtype=np.float64))
            pet_affine = pet_nii.affine
            pet_header = pet_nii.header
            voxel_size = tuple(pet_header.get_zooms()[:3])
        except Exception as e:
            raise RuntimeError(f"Failed to load PET image: {e}")

        # Load MR/CT mask for anatomical guidance (optional)
        mr_data = None
        if mask_path is not None:
            try:
                mask_nii = nib.load(str(mask_path))
                mr_data = np.asarray(mask_nii.get_fdata(dtype=np.float64))
            except Exception as e:
                raise RuntimeError(f"Failed to load mask/MR image: {e}")

        # Handle 4D data (sum over time)
        if pet_data.ndim == 4:
            print("[PLS-PVC] 4D data detected, summing over time axis...")
            pet_data = np.sum(pet_data, axis=-1)

        # Run PLS-PVC (MATLAB port with CuPy/GPU)
        try:
            corrected_data = pls_pvc_3d(
                pet=pet_data,
                mr=mr_data,
                fwhm=fwhm_xyz,
                voxel_size=voxel_size,
                niter=iterations,
                mu=pls_mu,
                use_mr_like_mask=pls_use_mr_like_mask,
                threshold_percentile=pls_threshold_percentile,
                use_gpu=True,  # Auto-detects CuPy
                verbose=True
            )

            # Save corrected image
            corrected_nii = nib.Nifti1Image(corrected_data, affine=pet_affine, header=pet_header)
            nib.save(corrected_nii, str(out_path))

            return str(out_path)

        except Exception as e:
            raise RuntimeError(f"PLS-PVC execution failed: {e}")

    # Import petpvc wrapper (for non-PLS methods)
    try:
        from unified_pvc.methods.petpvc_wrapper import PETPVCWrapper
    except ImportError as e:
        # Try alternative import paths
        try:
            from methods.petpvc_wrapper import PETPVCWrapper
        except ImportError:
            raise RuntimeError(
                f"Could not import PETPVCWrapper from unified_pvc.\n"
                f"Error: {e}\n"
                f"sys.path: {sys.path}\n"
                f"Searched locations:\n"
                + "\n".join(f"  - {p}" for p in _possible_paths) +
                f"\nEnsure unified_pvc package is installed or reachable."
            )

    # Load input image
    try:
        pet_nii = nib.load(str(pet_path))
        pet_data = np.asarray(pet_nii.get_fdata(dtype=np.float32))
        pet_affine = pet_nii.affine
        pet_header = pet_nii.header
    except Exception as e:
        raise RuntimeError(f"Failed to load PET image: {e}")

    # Load mask if provided
    mask_data = None
    if mask_path is not None:
        try:
            mask_nii = nib.load(str(mask_path))
            mask_data = np.asarray(mask_nii.get_fdata(dtype=np.float32))
        except Exception as e:
            raise RuntimeError(f"Failed to load mask image: {e}")

    # Create petpvc wrapper and run correction
    try:
        # Build petpvc kwargs - only core parameters supported by wrapper
        petpvc_kwargs = {
            "method": method,
            "fwhm": fwhm_xyz,  # petpvc expects (fwhm_x, fwhm_y, fwhm_z)
            "environment": os.environ.copy(),  # Pass current environment to subprocess
        }

        # Handle method-specific iteration parameters
        # IY (Iterative Yang) uses -n flag
        # RL, VC, GTM, RBV, MG, MTC, STC, LABBE use -k flag
        if method == "IY":
            petpvc_kwargs["iterations"] = iterations  # -n flag for IY
        else:
            # Default: use deconvolution_iterations if explicitly provided, otherwise use iterations
            if deconvolution_iterations is not None:
                petpvc_kwargs["iterations"] = deconvolution_iterations
            else:
                petpvc_kwargs["iterations"] = iterations

        corrector = PETPVCWrapper(**petpvc_kwargs)
        
        # Build petpvc command-line for advanced parameters (alpha, stopping criterion, etc.)
        # These will be passed directly via CLI since wrapper doesn't support them
        additional_cli_args = []
        if alpha is not None:
            additional_cli_args.extend(["-a", str(alpha)])
        if stopping_criterion is not None:
            additional_cli_args.extend(["-s", str(stopping_criterion)])
        if disable_non_negativity:
            additional_cli_args.append("-0")
        if debug:
            additional_cli_args.append("-d")
        
        # If we have advanced parameters, we need to use petpvc directly instead of the wrapper
        if additional_cli_args:
            # Use petpvc CLI directly with all parameters
            # Create temporary directory for petpvc operations
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Save input image temporarily
                input_temp = tmpdir_path / "pet_input.nii.gz"
                pet_nii_temp = nib.Nifti1Image(pet_data, affine=pet_affine, header=pet_header)
                nib.save(pet_nii_temp, str(input_temp))
                
                # Save mask temporarily if provided
                mask_temp = None
                if mask_data is not None:
                    mask_temp = tmpdir_path / "mask_input.nii.gz"
                    mask_nii_temp = nib.Nifti1Image(mask_data, affine=pet_affine)
                    nib.save(mask_nii_temp, str(mask_temp))
                
                # Build petpvc CLI command
                petpvc_cmd = [
                    "petpvc",
                    "-i", str(input_temp),
                    "-o", str(out_path),
                    "-p", method,
                    "-x", str(fwhm_xyz[0]),
                    "-y", str(fwhm_xyz[1]),
                    "-z", str(fwhm_xyz[2]),
                ]
                
                # Add iterations
                if method == "IY":
                    petpvc_cmd.extend(["-n", str(iterations)])
                else:
                    petpvc_cmd.extend(["-k", str(iterations)])
                
                # Add mask if provided
                if mask_temp is not None:
                    petpvc_cmd.extend(["-m", str(mask_temp)])
                
                # Add advanced parameters
                petpvc_cmd.extend(additional_cli_args)
                
                # Execute petpvc
                result = subprocess.run(
                    petpvc_cmd,
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"petpvc failed: {result.stderr}")
        else:
            # Use wrapper for standard parameters (no advanced options)
            # Run 3D or 4D correction depending on data shape
            if pet_data.ndim == 3:
                corrected_data = corrector.correct_3d(
                    pet_data=pet_data,
                    pet_affine=pet_affine,
                    pet_header=pet_header,
                    mask=mask_data,
                    auto_mask=True
                )
            elif pet_data.ndim == 4:
                corrected_data = corrector.correct_4d(
                    pet_data=pet_data,
                    pet_affine=pet_affine,
                    pet_header=pet_header,
                    mask=mask_data,
                    auto_mask=True
                )
            else:
                raise ValueError(f"Unsupported PET data shape: {pet_data.shape}")
            
            # Save corrected image
            corrected_nii = nib.Nifti1Image(corrected_data, affine=pet_affine, header=pet_header)
            nib.save(corrected_nii, str(out_path))

    except Exception as e:
        raise RuntimeError(f"PVC execution failed: {e}")

    return str(out_path)


# Convenience function for batch processing
def run_pvc_batch(
    pet_image_paths: List[str],
    output_dir: str,
    methods: List[str] = None,
    fwhm_xyz: Tuple[float, float, float] = (0.849, 0.798, 1.023),
    iterations_list: List[int] = None,
    mask_path: Optional[str] = None,
) -> dict:
    """
    Run PVC on multiple images with multiple methods and iteration counts.

    Parameters
    ----------
    pet_image_paths : List[str]
        List of paths to input PET images
    output_dir : str
        Directory where outputs will be saved
    methods : List[str]
        List of PVC methods to apply. Default: ["RL"]
    fwhm_xyz : Tuple[float, float, float]
        PSF FWHM in (x, y, z) mm
    iterations_list : List[int]
        List of iteration counts to test. Default: [10]
    mask_path : Optional[str]
        Single mask for all images (or None for auto-mask)

    Returns
    -------
    dict
        Dictionary mapping (method, iterations) -> output_path
    """
    if methods is None:
        methods = ["RL"]
    if iterations_list is None:
        iterations_list = [10]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for pet_path in pet_image_paths:
        pet_base = Path(pet_path).stem
        for method in methods:
            for n_iter in iterations_list:
                out_name = f"{pet_base}_PVC_{method}_iter{n_iter}.nii.gz"
                out_path = str(output_dir_path / out_name)

                try:
                    result_path = run_pvc(
                        pet_image_path=pet_path,
                        output_path=out_path,
                        method=method,
                        fwhm_xyz=fwhm_xyz,
                        iterations=n_iter,
                        mask_path=mask_path,
                    )
                    results[(method, n_iter)] = result_path
                except Exception as e:
                    print(f"Warning: PVC failed for {pet_base} with {method} (iter={n_iter}): {e}")

    return results
