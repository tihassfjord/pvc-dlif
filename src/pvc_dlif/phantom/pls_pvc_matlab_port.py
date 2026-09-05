"""
Parallel Level Sets (PLS) Partial Volume Correction - MATLAB Port
===================================================================

Faithful Python port of the MATLAB PLS-PVC implementation using CuPy for GPU acceleration.

This implementation uses the Split Bregman method to solve the PVC optimization problem
with anatomical guidance from MR/CT images.

Reference:
- Original MATLAB implementation: master/matlab_pvc/PVC_3D/
- Algorithm: Split Bregman optimization with PLS regularization
- Paper: Zhu et al., "Deconvolution-based partial volume correction with
  parallel level set regularization", PMB 2021

Mathematical Formulation:
-------------------------
Minimize: (μ/2)||H*u - g||²₂ + (λ/2)||B*∇u||₁

Where:
- g: observed PET image
- u: corrected PET image (to solve for)
- H: convolution with PSF (forward model)
- B = I - v*v^T: PLS operator (v = normalized MR gradient)
- μ: data fidelity weight
- λ: regularization weight (adaptive)

Algorithm: Split Bregman
-------------------------
1. u-subproblem: gradient descent with line search
2. d-subproblem: soft thresholding (shrinkage)
3. b-update: Bregman parameter update
4. λ-update: adaptive residual balancing

GPU Acceleration:
-----------------
Uses CuPy for all array operations. Falls back to NumPy if CUDA not available.

Author: Ported from MATLAB to Python with CuPy
Date: November 2024
"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[PLS-PVC] CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    print("[PLS-PVC] CuPy not available - using CPU (NumPy)")

from typing import Tuple, List, Optional, Union
import warnings


def get_array_module(arr):
    """Get the appropriate array module (cupy or numpy) for an array."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_device(arr, use_gpu=True):
    """Move array to GPU if available and requested, otherwise CPU."""
    if use_gpu and CUPY_AVAILABLE:
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr
    else:
        if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr


def to_cpu(arr):
    """Move array to CPU."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def create_mr_like_mask(
    pet: np.ndarray,
    threshold_percentile: float = 30.0,
    smooth_sigma: float = 2.0,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Create an MR-like anatomical mask from PET image using simple thresholding.

    This creates a pseudo-anatomical image to guide PLS-PVC when no actual
    MR/CT is available. The mask uses percentile-based thresholding to create
    a soft tissue-like anatomical guide.

    Parameters
    ----------
    pet : np.ndarray
        Input PET image (3D)
    threshold_percentile : float, optional
        Percentile threshold for foreground/background separation
        Lower values = more aggressive background removal
        Typical range: 20-50
        Default: 30.0 (30th percentile)
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in voxels) to reduce noise
        Default: 2.0
    use_gpu : bool, optional
        Use GPU for computations if available
        Default: True

    Returns
    -------
    np.ndarray
        MR-like mask (same shape as PET, values normalized 0-1)

    Examples
    --------
    >>> import numpy as np
    >>> pet = np.random.rand(64, 64, 64) * 1000
    >>> mr_like = create_mr_like_mask(pet, threshold_percentile=25)
    >>> print(mr_like.shape, mr_like.min(), mr_like.max())
    (64, 64, 64) 0.0 1.0
    """
    # Move to GPU if requested
    pet_proc = to_device(pet, use_gpu)
    xp = get_array_module(pet_proc)

    # Smooth to reduce noise
    if xp is cp:
        from cupyx.scipy import ndimage as ndi
    else:
        from scipy import ndimage as ndi

    pet_smooth = ndi.gaussian_filter(pet_proc, sigma=smooth_sigma)

    # Simple percentile thresholding
    thresh = xp.percentile(pet_smooth, threshold_percentile)

    # Create soft mask above threshold
    # Scale values above threshold to [0, 1]
    mask = xp.clip((pet_smooth - thresh) / (xp.max(pet_smooth) - thresh + 1e-10), 0, 1)

    # Smooth the mask to create gradual transitions
    # This creates edge gradients similar to MR
    mask_smooth = ndi.gaussian_filter(mask, sigma=smooth_sigma * 0.5)

    # Ensure normalized to [0, 1]
    mask_smooth = mask_smooth - xp.min(mask_smooth)
    mask_max = xp.max(mask_smooth)
    if mask_max > 0:
        mask_smooth = mask_smooth / mask_max

    # Move back to CPU
    mask_out = to_cpu(mask_smooth)

    return mask_out


class PLSPVCCorrector:
    """
    3D Partial Volume Correction using Parallel Level Sets (PLS) method.

    This is a faithful port of the MATLAB implementation with GPU acceleration via CuPy.

    Parameters
    ----------
    fwhm : tuple of float
        System FWHM in (x, y, z) dimensions (unit: mm)
    voxel_size : tuple of float
        Voxel size in (x, y, z) dimensions (unit: mm)
    mu : float, optional
        Data fidelity regularization parameter (default: 17)
        Higher mu = stronger data fidelity (closer to original image)
    niter : int, optional
        Maximum number of iterations (default: 100)
    use_gpu : bool, optional
        Use GPU acceleration if available (default: True)
    verbose : bool, optional
        Print progress information (default: True)

    Attributes
    ----------
    lambda_init : float
        Initial lambda value (68, from MATLAB)
    mr_smooth : float
        Smoothing parameter to avoid zero in MR gradient (6e-4, from MATLAB)
    eta : float
        Decay parameter for average objective (0.995, from MATLAB)
    epsilon_factor : float
        Stopping tolerance factor (0.1, from MATLAB)
    beta : float
        Residual balancing threshold (8, from MATLAB)
    alpha_max : float
        Maximum lambda update ratio (100, from MATLAB)
    """

    def __init__(
        self,
        fwhm: Tuple[float, float, float],
        voxel_size: Tuple[float, float, float],
        mu: float = 17.0,
        niter: int = 100,
        use_gpu: bool = True,
        verbose: bool = True
    ):
        self.fwhm = fwhm
        self.voxel_size = voxel_size
        self.mu = mu
        self.niter = niter
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.verbose = verbose

        # MATLAB parameters - DO NOT CHANGE (for exact equivalence)
        self.lambda_init = 68.0  # Initial lambda (MATLAB: lambdainit=68)
        self.mr_smooth = 6e-4    # MR smoothing (MATLAB: MR_smooth=6e-4)
        self.eta = 0.995         # Average objective decay (MATLAB: eta=0.995)
        self.epsilon_factor = 0.1  # Stopping tolerance factor (MATLAB: 0.1)
        self.beta = 8.0          # Residual balancing threshold (MATLAB: beta=8)
        self.alpha_max = 100.0   # Max lambda update ratio (MATLAB: alpha_max=100)

        # Line search parameters - DO NOT CHANGE
        self.ls_epsilon = 1e-4   # Line search tolerance (MATLAB: epsilon=1e-4)
        self.ls_rho = 0.4        # Line search reduction (MATLAB: rho=0.4)
        self.ls_max_iter = 8     # Max line search iterations (MATLAB: 8)

        # Convergence parameters - DO NOT CHANGE
        self.conv_threshold = 1e-3  # Convergence threshold (MATLAB: 1e-3)
        self.stop_counter_thresh = 3  # Stop counter threshold (MATLAB: 3)

        # Select array module
        self.xp = cp if self.use_gpu else np

        if self.verbose:
            mode = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
            print(f"[PLS-PVC] Initialized in {mode} mode")
            print(f"[PLS-PVC] FWHM: {fwhm} mm")
            print(f"[PLS-PVC] Voxel size: {voxel_size} mm")
            print(f"[PLS-PVC] μ={mu}, max_iter={niter}")

    def _create_psf_3d(self) -> np.ndarray:
        """
        Create 3D Gaussian PSF kernel.

        Exactly matches MATLAB implementation in PVC_3D.m lines 13-25.

        Returns
        -------
        np.ndarray
            3D PSF kernel (normalized to sum=1)
        """
        # Convert FWHM from mm to grid units (MATLAB: lines 14-16)
        fwhm_x = self.fwhm[0] / self.voxel_size[0]
        fwhm_y = self.fwhm[1] / self.voxel_size[1]
        fwhm_z = self.fwhm[2] / self.voxel_size[2]

        # Convert FWHM to sigma (MATLAB: lines 18-20)
        sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
        sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2)))

        # Create 3D grid (MATLAB: lines 21-22)
        # MATLAB uses 1-indexed 256x256x256 grid
        x = np.arange(1, 257)
        y = np.arange(1, 257)
        z = np.arange(1, 257)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Gaussian PSF formula (MATLAB: lines 23-24)
        # Note: MATLAB meshgrid uses (y,x,z) order, but indexing='ij' matches
        norm_factor = 1.0 / ((2 * np.pi) ** (3/2) * sigma_x * sigma_y * sigma_z)
        psf = norm_factor * np.exp(-((X - 128) / sigma_x) ** 2 / 2) * \
                           np.exp(-((Y - 128) / sigma_y) ** 2 / 2) * \
                           np.exp(-((Z - 128) / sigma_z) ** 2 / 2)

        # Extract central region (MATLAB: line 25)
        # MATLAB: psf(122:134,122:134,122:134) is 13x13x13
        psf = psf[121:134, 121:134, 121:134]  # Python 0-indexed

        # Normalize (MATLAB: line 25)
        psf = psf / np.sum(psf)

        return psf

    def _symmetric_pad(self, arr):
        """
        Symmetrically pad 3D array.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 113-122.

        Parameters
        ----------
        arr : array
            Input 3D array

        Returns
        -------
        array
            Padded array (each dimension increased by 2)
        """
        xp = get_array_module(arr)
        H, L, W = arr.shape

        # Create padded array (MATLAB: line 117)
        out = xp.zeros((H + 2, L + 2, W + 2), dtype=arr.dtype)

        # Copy original data (MATLAB: line 118)
        out[1:-1, 1:-1, 1:-1] = arr

        # Symmetric padding (MATLAB: lines 119-121)
        out[0, :, :] = out[2, :, :]
        out[-1, :, :] = out[-3, :, :]
        out[:, 0, :] = out[:, 2, :]
        out[:, -1, :] = out[:, -3, :]
        out[:, :, 0] = out[:, :, 2]
        out[:, :, -1] = out[:, :, -3]

        return out

    def _gradient_3d(self, x, choice: str = 'forward'):
        """
        Compute 3D gradient using finite differences.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 124-139.

        Parameters
        ----------
        x : array
            Input 3D array
        choice : str
            'forward' or 'backward' difference

        Returns
        -------
        list of array
            [gx_x, gx_y, gx_z] gradient components
        """
        xp = get_array_module(x)

        # Create finite difference mask (MATLAB: lines 128-132)
        if choice == 'forward':
            mask = xp.array([0, -1, 1], dtype=x.dtype)
        elif choice == 'backward':
            mask = xp.array([-1, 1, 0], dtype=x.dtype)
        else:
            raise ValueError(f"Invalid choice: {choice}")

        # Symmetric padding (MATLAB: line 134)
        X = self._symmetric_pad(x)

        # Compute gradients via convolution (MATLAB: lines 135-137)
        # x-direction
        mask_x = mask.reshape(3, 1, 1)
        gx_x = self._convolve_3d(X, mask_x, mode='same')
        gx_x = gx_x[1:-1, 1:-1, 1:-1]

        # y-direction
        mask_y = mask.reshape(1, 3, 1)
        gx_y = self._convolve_3d(X, mask_y, mode='same')
        gx_y = gx_y[1:-1, 1:-1, 1:-1]

        # z-direction
        mask_z = mask.reshape(1, 1, 3)
        gx_z = self._convolve_3d(X, mask_z, mode='same')
        gx_z = gx_z[1:-1, 1:-1, 1:-1]

        return [gx_x, gx_y, gx_z]

    def _convolve_3d(self, arr, kernel, mode='same'):
        """
        3D convolution wrapper (handles CuPy/NumPy).

        Parameters
        ----------
        arr : array
            Input array
        kernel : array
            Convolution kernel
        mode : str
            Convolution mode ('same', 'valid', 'full')

        Returns
        -------
        array
            Convolved array
        """
        xp = get_array_module(arr)

        if xp is cp:
            # CuPy doesn't have convolve, use scipy.ndimage or manual FFT
            from cupyx.scipy import ndimage as ndi
            return ndi.convolve(arr, kernel, mode='constant')
        else:
            from scipy import ndimage as ndi
            return ndi.convolve(arr, kernel, mode='constant')

    def _operator_pls(self, gv: List, x: List):
        """
        Apply PLS operator: B = (I - gv*gv^T).

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 141-149.

        The PLS operator projects gradients onto the orthogonal complement of
        the MR gradient direction, preserving edges aligned with anatomy.

        Parameters
        ----------
        gv : list of array
            Normalized MR gradient [gv_x, gv_y, gv_z]
        x : list of array
            Input gradient [x_x, x_y, x_z]

        Returns
        -------
        list of array
            B*x = (I - gv*gv^T)*x
        """
        # MATLAB: lines 146-148
        # B_out{1} = (1-gv{1}.^2).*in{1} - gv{1}.*gv{2}.*in{2} - gv{1}.*gv{3}.*in{3}
        # B_out{2} = -gv{1}.*gv{2}.*in{1} + (1-gv{2}.^2).*in{2} - gv{2}.*gv{3}.*in{3}
        # B_out{3} = -gv{1}.*gv{3}.*in{1} - gv{2}.*gv{3}.*in{2} + (1-gv{3}.^2).*in{3}

        B_out = []
        B_out.append((1 - gv[0]**2) * x[0] - gv[0] * gv[1] * x[1] - gv[0] * gv[2] * x[2])
        B_out.append(-gv[0] * gv[1] * x[0] + (1 - gv[1]**2) * x[1] - gv[1] * gv[2] * x[2])
        B_out.append(-gv[0] * gv[2] * x[0] - gv[1] * gv[2] * x[1] + (1 - gv[2]**2) * x[2])

        return B_out

    def _d_pls_update(self, B_gu: List, b: List, lambda_val: float):
        """
        Solve d-subproblem using soft thresholding.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 151-159.

        Parameters
        ----------
        B_gu : list of array
            B operator applied to gradient of u
        b : list of array
            Bregman parameters
        lambda_val : float
            Regularization parameter

        Returns
        -------
        list of array
            Updated d values
        """
        xp = get_array_module(B_gu[0])

        # MATLAB: line 153
        # s = sqrt((B_gu{1}+b{1}).^2 + (B_gu{2}+b{2}).^2 + (B_gu{3}+b{3}).^2)
        s = xp.sqrt((B_gu[0] + b[0])**2 + (B_gu[1] + b[1])**2 + (B_gu[2] + b[2])**2)

        # MATLAB: line 154
        # tmp = s - 1/lambda; tmp(tmp<0) = 0
        tmp = s - 1.0 / lambda_val
        tmp = xp.maximum(tmp, 0)

        # MATLAB: lines 156-158
        # d_out{i} = tmp .* (B_gu{i}+b{i}) ./ (s+eps)
        d_out = []
        eps = xp.finfo(xp.float64).eps
        for i in range(3):
            d_out.append(tmp * (B_gu[i] + b[i]) / (s + eps))

        return d_out

    def _b_update(self, B_gu: List, d: List, b: List):
        """
        Update Bregman parameters.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 161-163.

        Parameters
        ----------
        B_gu : list of array
            B operator applied to gradient of u
        d : list of array
            Current d values
        b : list of array
            Current Bregman parameters

        Returns
        -------
        list of array
            Updated Bregman parameters
        """
        # MATLAB: line 163
        # b_out = cellfun(@(x,y,z)(x+y-z), b, B_gu, d, 'un', false)
        b_out = []
        for i in range(3):
            b_out.append(b[i] + B_gu[i] - d[i])

        return b_out

    def _lambda_update(
        self,
        d: List,
        d_old: List,
        lambda_val: float,
        gv: List,
        B_gu: List,
        b: List,
        flag: int
    ) -> Tuple[float, float, float]:
        """
        Update lambda adaptively with residual balancing.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 166-220.

        Parameters
        ----------
        d : list of array
            Current d values
        d_old : list of array
            Previous d values
        lambda_val : float
            Current lambda
        gv : list of array
            Normalized MR gradient
        B_gu : list of array
            B operator applied to gradient of u
        b : list of array
            Bregman parameters
        flag : int
            Control flag for adaptive vs fixed lambda

        Returns
        -------
        tuple
            (updated lambda, normalized primal residual, normalized dual residual)
        """
        xp = get_array_module(d[0])

        # Primal residual (MATLAB: lines 169-177)
        r_prim = [B_gu[i] - d[i] for i in range(3)]
        r_prim_norms = [xp.linalg.norm(r.ravel()) for r in r_prim]
        r_prim = xp.linalg.norm(xp.array(r_prim_norms))

        factor1_norms = [xp.linalg.norm(B_gu[i].ravel()) for i in range(3)]
        factor1 = xp.linalg.norm(xp.array(factor1_norms))

        factor2_norms = [xp.linalg.norm(d[i].ravel()) for i in range(3)]
        factor2 = xp.linalg.norm(xp.array(factor2_norms))

        r_prim_norm = float(r_prim / max(factor1, factor2))

        # Dual residual (MATLAB: lines 179-197)
        d_tmp = [d_old[i] - d[i] for i in range(3)]
        B_dd = self._operator_pls(gv, d_tmp)

        r_dual1 = self._gradient_3d(B_dd[0], 'backward')
        r_dual2 = self._gradient_3d(B_dd[1], 'backward')
        r_dual3 = self._gradient_3d(B_dd[2], 'backward')
        r_dual = [r_dual1[0], r_dual2[1], r_dual3[2]]

        r_dual = [lambda_val * r for r in r_dual]
        r_dual_norms = [xp.linalg.norm(r.ravel()) for r in r_dual]
        r_dual = xp.linalg.norm(xp.array(r_dual_norms))

        B_gb = self._operator_pls(gv, b)
        factor1 = self._gradient_3d(B_gb[0], 'backward')
        factor2 = self._gradient_3d(B_gb[1], 'backward')
        factor3 = self._gradient_3d(B_gb[2], 'backward')
        factor = [factor1[0], factor2[1], factor3[2]]

        factor_norms = [xp.linalg.norm(f.ravel()) for f in factor]
        factor = xp.linalg.norm(xp.array(factor_norms))

        r_dual_norm = float(r_dual / factor)

        # Compute update ratio (MATLAB: lines 199-207)
        alpha_tmp = (r_prim_norm / r_dual_norm) ** 0.5

        if alpha_tmp >= 1 and alpha_tmp < self.alpha_max:
            alpha = alpha_tmp
        elif alpha_tmp > 1 / self.alpha_max and alpha_tmp < 1:
            alpha = 1.0 / alpha_tmp
        else:
            alpha = self.alpha_max

        # Update lambda (MATLAB: lines 209-218)
        if flag == 0:
            if r_prim > self.beta * r_dual:
                lambda_out = alpha * lambda_val
            elif r_dual > self.beta * r_prim:
                lambda_out = (1.0 / alpha) * lambda_val
            else:
                lambda_out = lambda_val
        else:
            lambda_out = self.lambda_init

        return lambda_out, r_prim_norm, r_dual_norm

    def _objective_function(
        self,
        imgy,
        u_vec,
        gv: List,
        psf,
        d: List,
        b: List,
        mu: float,
        lambda_val: float
    ) -> Tuple[float, np.ndarray]:
        """
        Compute objective function and its gradient for u-subproblem.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 68-89.

        Parameters
        ----------
        imgy : array
            Observed PET image
        u_vec : array
            Current u estimate (flattened)
        gv : list of array
            Normalized MR gradient
        psf : array
            PSF kernel
        d : list of array
            Current d values
        b : list of array
            Bregman parameters
        mu : float
            Data fidelity parameter
        lambda_val : float
            Regularization parameter

        Returns
        -------
        tuple
            (objective value, gradient vector)
        """
        xp = get_array_module(imgy)

        # Reshape u (MATLAB: line 70)
        img_u = u_vec.reshape(imgy.shape)

        # Create flipped PSF for adjoint (MATLAB: line 71)
        # MATLAB: psf_neg = flip(rot90(psf,2),3)
        # rot90(psf, 2) = rotate 180 degrees in x-y plane
        # flip(..., 3) = flip along z-axis
        psf_neg = xp.rot90(psf, 2, axes=(0, 1))
        psf_neg = xp.flip(psf_neg, axis=2)

        # Compute gradient of u (MATLAB: line 72)
        gu = self._gradient_3d(img_u, 'forward')

        # Objective function value (MATLAB: lines 74-77)
        B_gu = self._operator_pls(gv, gu)

        # Data fidelity term
        fx1 = self._convolve_3d(img_u, psf, mode='same') - imgy

        # Regularization term
        fx2_components = [(d[i] - B_gu[i] - b[i])**2 for i in range(3)]
        fx2 = sum(xp.sum(comp) for comp in fx2_components)

        fx = float((mu / 2) * xp.sum(fx1**2) + (lambda_val / 2) * fx2)

        # Gradient of objective function (MATLAB: lines 79-88)
        cell_fx2 = [d[i] - B_gu[i] - b[i] for i in range(3)]
        B_fx2 = self._operator_pls(gv, cell_fx2)

        dfx1 = self._convolve_3d(
            self._convolve_3d(img_u, psf, mode='same') - imgy,
            psf_neg,
            mode='same'
        )

        dfx2_tmp1 = self._gradient_3d(B_fx2[0], 'backward')
        dfx2_tmp2 = self._gradient_3d(B_fx2[1], 'backward')
        dfx2_tmp3 = self._gradient_3d(B_fx2[2], 'backward')
        dfx2 = [dfx2_tmp1[0], dfx2_tmp2[1], dfx2_tmp3[2]]

        dfx = mu * dfx1 + lambda_val * sum(dfx2)
        dfx = dfx.ravel()

        return fx, dfx

    def _line_search(
        self,
        fun,
        x,
        x_old,
        C: float
    ) -> float:
        """
        Compute step size using line search with Armijo condition.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 91-111.

        Parameters
        ----------
        fun : callable
            Objective function returning (value, gradient)
        x : array
            Current iterate
        x_old : array
            Previous iterate
        C : float
            Average objective value

        Returns
        -------
        float
            Step size alpha
        """
        xp = get_array_module(x)

        # Compute gradients (MATLAB: line 97)
        _, dfx = fun(x)
        _, dfx_old = fun(x_old)

        # Barzilai-Borwein step size (MATLAB: lines 98-99)
        s = x - x_old
        y = dfx - dfx_old

        alpha = float((s @ y) / (y @ y))

        # Armijo line search (MATLAB: lines 102-109)
        test_step = x - alpha * dfx

        for i in range(self.ls_max_iter):
            f_test, _ = fun(test_step)

            # Armijo condition (MATLAB: line 105)
            if f_test <= C - self.ls_epsilon * alpha * (dfx @ dfx):
                return alpha

            # Reduce step size (MATLAB: line 108)
            alpha = alpha * self.ls_rho
            test_step = x - alpha * dfx

        if self.verbose:
            print(f"  [Line search] Max iterations ({self.ls_max_iter}) reached")

        return alpha

    def correct_3d(
        self,
        pet: np.ndarray,
        mr: Optional[np.ndarray] = None,
        use_mr_like_mask: bool = False,
        threshold_percentile: float = 30.0,
        verbose: Optional[bool] = None
    ) -> np.ndarray:
        """
        Perform 3D partial volume correction.

        Exactly matches MATLAB implementation in PLS_SplitBregman_3D.m lines 1-66.

        Parameters
        ----------
        pet : np.ndarray
            Input 3D PET image
        mr : np.ndarray, optional
            Input 3D MR/CT image for anatomical guidance
            If None, uses self-guided mode (MR = PET or MR-like mask)
        use_mr_like_mask : bool, optional
            If True and mr is None, creates an MR-like anatomical mask from PET
            using thresholding instead of pure self-guided mode.
            Can improve edge preservation compared to self-guided.
            Default: False
        threshold_percentile : float, optional
            Percentile threshold for MR-like mask generation (20-50 range)
            Only used if use_mr_like_mask=True and mr=None
            Default: 30.0
        verbose : bool, optional
            Override instance verbose setting

        Returns
        -------
        np.ndarray
            Corrected PET image (same shape as input, CPU array)
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print("\n" + "="*70)
            print("PLS-PVC: 3D Partial Volume Correction")
            print("="*70)

        # Handle self-guided mode
        if mr is None:
            if use_mr_like_mask:
                if verbose:
                    print(f"[PLS-PVC] Creating MR-like mask from PET (threshold={threshold_percentile}%)")
                mr = create_mr_like_mask(
                    pet,
                    threshold_percentile=threshold_percentile,
                    smooth_sigma=2.0,
                    use_gpu=self.use_gpu
                )
                if verbose:
                    print(f"[PLS-PVC] MR-like mask created (range: [{mr.min():.3f}, {mr.max():.3f}])")
            else:
                if verbose:
                    print("[PLS-PVC] No MR provided - using self-guided mode (MR = PET)")
                mr = pet.copy()

        # Move to GPU if requested
        pet = to_device(pet, self.use_gpu)
        mr = to_device(mr, self.use_gpu)
        xp = get_array_module(pet)

        # Create PSF (on CPU, then move to GPU if needed)
        psf = self._create_psf_3d()
        psf = to_device(psf, self.use_gpu)

        if verbose:
            print(f"[PLS-PVC] Input shape: {pet.shape}")
            print(f"[PLS-PVC] PSF shape: {psf.shape}")
            print(f"[PLS-PVC] Device: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")

        # Rescale (MATLAB: lines 10-14)
        scale_factor_pet = float(xp.max(pet))
        mr = mr / float(xp.max(mr))
        pet = pet / scale_factor_pet

        if verbose:
            print(f"[PLS-PVC] Scale factor: {scale_factor_pet:.2e}")

        # Initialize (MATLAB: lines 17-21)
        init = xp.zeros(pet.shape, dtype=xp.float64)
        d = [init.copy() for _ in range(3)]
        b = [init.copy() for _ in range(3)]
        u_current = pet.copy()
        u_old = xp.zeros(pet.shape, dtype=xp.float64)
        u = pet.copy()

        # Compute normalized MR gradient (MATLAB: lines 22-24)
        gv = self._gradient_3d(mr, 'forward')
        gv_norm = xp.sqrt(gv[0]**2 + gv[1]**2 + gv[2]**2 + self.mr_smooth**2)
        gv = [gv[i] / gv_norm for i in range(3)]

        # Initialize adaptive parameters (MATLAB: line 25)
        P = 1.0
        lambda_val = self.lambda_init

        # Initial objective
        C, _ = self._objective_function(
            pet, pet.ravel(), gv, psf, d, b, self.mu, lambda_val
        )

        # Stopping tolerance (MATLAB: line 29)
        # Normalized to phantom size 181x210x181
        epsilon = self.epsilon_factor * pet.size / (181 * 210 * 181)

        # Main iteration (MATLAB: lines 30-64)
        stop_counter = 0
        flag = 0

        if verbose:
            print("\n[PLS-PVC] Starting iterations...")
            print("-" * 70)

        for iteration in range(1, self.niter + 1):
            # U-subproblem (MATLAB: lines 32-36)
            u_current = u.copy()

            fun = lambda x_vec: self._objective_function(
                pet, x_vec, gv, psf, d, b, self.mu, lambda_val
            )

            _, dfx = fun(u_current.ravel())
            alpha = self._line_search(fun, u_current.ravel(), u_old.ravel(), C)
            u = u_current - alpha * dfx.reshape(u_current.shape)

            # D-subproblem (MATLAB: lines 38-41)
            gu = self._gradient_3d(u, 'forward')
            B_gu = self._operator_pls(gv, gu)
            d_old = [d[i].copy() for i in range(3)]
            d = self._d_pls_update(B_gu, b, lambda_val)

            # Update b (MATLAB: line 43)
            b = self._b_update(B_gu, d, b)

            # Update lambda (MATLAB: line 45)
            lambda_val, rp, rd = self._lambda_update(
                d, d_old, lambda_val, gv, B_gu, b, flag
            )

            # Update parameters for gradient descent (MATLAB: lines 48-50)
            fx, _ = self._objective_function(
                pet, u.ravel(), gv, psf, d, b, self.mu, lambda_val
            )
            P_new = self.eta * P + 1
            C = (self.eta * P * C + fx) / P_new
            P = P_new
            u_old = u_current.copy()

            # Compute relative change (MATLAB: line 51)
            rel_change = float(xp.linalg.norm((u - u_current).ravel()) /
                             xp.linalg.norm(u_current.ravel()))

            if verbose:
                print(f"[Iter {iteration:3d}] Change: {rel_change*100:6.3f}% | "
                      f"λ: {lambda_val:8.2e} | rp: {rp:8.2e} | rd: {rd:8.2e}")

            # Stopping criteria (MATLAB: lines 53-63)
            if rel_change < self.conv_threshold:
                stop_counter += 1
            else:
                stop_counter = 0

            if stop_counter >= self.stop_counter_thresh and flag <= 1:
                flag += 1
                stop_counter = 0
                if verbose:
                    print(f"  → Switching to fixed λ mode")
            elif rp < epsilon and flag > 1:
                if verbose:
                    print(f"\n[PLS-PVC] Converged at iteration {iteration}")
                break

        # Rescale back (MATLAB: line 65)
        out = u * scale_factor_pet

        # Move back to CPU
        out = to_cpu(out)

        if verbose:
            print("-" * 70)
            print(f"[PLS-PVC] Completed after {iteration} iterations")
            print(f"[PLS-PVC] Output range: [{float(xp.min(out)):.2f}, {float(xp.max(out)):.2f}]")
            print("="*70 + "\n")

        return out


def pls_pvc_3d(
    pet: np.ndarray,
    mr: Optional[np.ndarray] = None,
    fwhm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    niter: int = 100,
    mu: float = 17.0,
    use_mr_like_mask: bool = False,
    threshold_percentile: float = 30.0,
    use_gpu: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Convenience function for 3D PLS-PVC.

    Exactly matches MATLAB functions PVC_3D.m and PVC_3D_wrapper.m.

    Parameters
    ----------
    pet : np.ndarray
        Input 3D PET image
    mr : np.ndarray, optional
        Input 3D MR/CT image for anatomical guidance
        If None, uses self-guided mode or MR-like mask
    fwhm : tuple of float
        System FWHM in (x, y, z) dimensions (mm)
    voxel_size : tuple of float
        Voxel size in (x, y, z) dimensions (mm)
    niter : int
        Maximum number of iterations
    mu : float
        Data fidelity parameter (default: 17)
    use_mr_like_mask : bool, optional
        If True and mr is None, creates MR-like mask from PET via thresholding
        Can improve edge preservation vs pure self-guided mode
        Default: False
    threshold_percentile : float, optional
        Percentile threshold for MR-like mask (20-50 range)
        Only used if use_mr_like_mask=True
        Default: 30.0
    use_gpu : bool
        Use GPU acceleration if available
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Corrected PET image

    Examples
    --------
    >>> import numpy as np
    >>> pet = np.random.rand(64, 64, 64)
    >>> # Standard self-guided
    >>> corrected = pls_pvc_3d(pet, fwhm=(2.0, 2.0, 2.0), voxel_size=(0.5, 0.5, 0.5))
    >>> # With MR-like mask for better edge preservation
    >>> corrected = pls_pvc_3d(pet, fwhm=(2.0, 2.0, 2.0), voxel_size=(0.5, 0.5, 0.5),
    ...                        use_mr_like_mask=True, threshold_percentile=25)
    """
    corrector = PLSPVCCorrector(
        fwhm=fwhm,
        voxel_size=voxel_size,
        mu=mu,
        niter=niter,
        use_gpu=use_gpu,
        verbose=verbose
    )

    return corrector.correct_3d(
        pet,
        mr,
        use_mr_like_mask=use_mr_like_mask,
        threshold_percentile=threshold_percentile,
        verbose=verbose
    )


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("PLS-PVC MATLAB Port - Test Script")
    print("="*70)

    # Create synthetic test data
    print("\n[Test] Creating synthetic phantom...")
    size = 64
    pet = np.random.rand(size, size, size).astype(np.float64) * 1000

    # Add hot spheres
    center = size // 2
    y, x, z = np.ogrid[:size, :size, :size]
    mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= (size/8)**2
    pet[mask] = 5000

    print(f"[Test] Phantom shape: {pet.shape}")
    print(f"[Test] Phantom range: [{pet.min():.1f}, {pet.max():.1f}]")

    # Run PLS-PVC
    print("\n[Test] Running PLS-PVC (self-guided mode)...")
    corrected = pls_pvc_3d(
        pet=pet,
        mr=None,  # Self-guided
        fwhm=(2.0, 2.0, 2.5),
        voxel_size=(1.0, 1.0, 1.0),
        niter=20,
        mu=17.0,
        use_gpu=True,
        verbose=True
    )

    print(f"\n[Test] Corrected shape: {corrected.shape}")
    print(f"[Test] Corrected range: [{corrected.min():.1f}, {corrected.max():.1f}]")
    print(f"[Test] Difference: {np.mean(np.abs(corrected - pet)):.1f}")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
