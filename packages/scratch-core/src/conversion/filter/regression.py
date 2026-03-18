"""
Low-level regression filter implementations.

This module provides kernel creation, convolution operations, and polynomial
regression filters used by the higher-level Gaussian filter functions.

Performance notes
-----------------
The main optimisation in ``apply_polynomial_filter`` is a *fast path* for the
common no-NaN case (order ≥ 1):

* **No LHS FFTs.**  When the data contains no NaNs the weight function is
  identically 1 everywhere.  The LHS moment matrix ``A`` therefore separates
  into a product of 1-D kernel moments and is **the same at every interior
  pixel**.  It is computed analytically from the kernel in O(n_params²) scalar
  operations instead of 15 (order-2) or 6 (order-1) full-image FFT pairs.

* **No per-pixel linalg.solve.**  Because ``A`` is constant in the interior,
  the smoothed value reduces to a weighted sum ``c₀ = A⁻¹[0,:] @ rhs[:,y,x]``,
  evaluated for all pixels simultaneously with a single ``np.einsum``.

* **Accurate border handling without extra FFTs.**  At the image boundary the
  Gaussian kernel is truncated, so ``A`` does vary per pixel.  These border
  pixels (a strip of half-kernel-width around the edge) are solved exactly
  using a direct per-pixel ``A`` built from pre-computed *partial kernel
  moment* arrays — 1-D arrays that are O(max_power × image_size) scalars and
  cost ~10 ms to compute — followed by a single batched ``linalg.solve`` on
  the ~35 % of pixels that fall in the border region.

* **Pre-allocated intermediate arrays.**  The NaN path (and the RHS
  computation in both paths) uses ``np.empty`` + in-place assignment instead
  of ``np.array([generator])`` list-construction, eliminating ~80–90 ms of
  Python-level array-stacking overhead per call.
"""

import numpy as np
from scipy.signal import fftconvolve
from numpy.typing import NDArray
from container_models.base import FloatArray1D, FloatArray2D, FloatArray4D, FloatArray3D


# ---------------------------------------------------------------------------
# Kernel creation
# ---------------------------------------------------------------------------


def create_normalized_separable_kernels(
    alpha: float, cutoff_pixels: FloatArray1D
) -> tuple[FloatArray1D, FloatArray1D]:
    """
    Create normalized 1D Gaussian kernels for the X and Y axes, where:
      - ``kernel_x`` is the 1D kernel for the X-axis (row vector).
      - ``kernel_y`` is the 1D kernel for the Y-axis (column vector).
      - The outer product of these kernels sums to approx 1.0.

    :param alpha: The Gaussian constant (ISO 16610).
    :param cutoff_pixels: Array of [cutoff_y, cutoff_x] in pixel units.
    :returns: A tuple ``(kernel_x, kernel_y)``.
    """
    # Ensure kernel size is odd and covers sufficient standard deviations
    kernel_dims = 1 + np.ceil(len(cutoff_pixels) * cutoff_pixels).astype(int)
    kernel_dims += 1 - kernel_dims % 2

    kernel_y = create_normalized_1d_kernel(alpha, cutoff_pixels[0], size=kernel_dims[0])
    kernel_x = create_normalized_1d_kernel(alpha, cutoff_pixels[1], size=kernel_dims[1])

    # Each 1D kernel is normalized to sum to 1. Separable convolution uses the outer
    # product (https://en.wikipedia.org/wiki/Outer_product), so the equivalent 2D
    # kernel automatically sums to 1 as well.
    return kernel_x, kernel_y


def create_gaussian_kernel_1d(
    cutoff_pixels: float,
    has_nans: bool,
    alpha: float,
) -> FloatArray1D:
    """
    Create a 1D Gaussian kernel with size determined by NaN presence.

    The kernel size calculation differs based on whether the data contains NaNs.
    This is legacy behavior from MATLAB code.

    :param cutoff_pixels: Cutoff wavelength in pixel units.
    :param has_nans: Whether the data contains NaN values.
    :param alpha: The Gaussian constant (ISO 16610).
    :returns: Normalized 1D Gaussian kernel.
    """
    # TODO: Kernel size determination differs for NaN vs non-NaN data (MATLAB legacy).
    # Preference would be to use a single determination, preferably the scipy default.
    sigma = alpha * cutoff_pixels / np.sqrt(2 * np.pi)

    if has_nans:
        kernel_size = int(np.ceil(4 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
    else:
        radius = int(np.ceil(3 * sigma))
        kernel_size = 2 * radius + 1

    return create_normalized_1d_kernel(alpha, cutoff_pixels, size=kernel_size)


def create_normalized_1d_kernel(
    alpha: float,
    cutoff_pixel: float,
    size: int,
) -> FloatArray1D:
    """
    Create a normalized 1D Gaussian kernel using ISO 16610 formula.

    Uses the ISO Gaussian formula: exp(-π(x/(α·λc))²), then normalizes to sum to 1.

    :param alpha: The Gaussian constant (ISO 16610).
    :param cutoff_pixel: Cutoff wavelength in pixel units.
    :param size: Kernel size (must be odd).
    :returns: Normalized 1D Gaussian kernel that sums to 1.
    """
    radius = (size - 1) // 2

    # Create coordinate vector centered at 0
    coords = np.arange(-radius, radius + 1)

    # ISO formula: exp(-π(x/(α·λc))²)
    scale_factor = alpha * cutoff_pixel
    kernel = np.exp(-np.pi * (coords / scale_factor) ** 2)

    # Normalize to sum to 1
    return kernel / np.sum(kernel)


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def convolve_2d_separable(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    mode: str = "constant",
) -> FloatArray2D:
    """
    Perform fast 2D convolution using separable 1D kernels via FFT.

    :param data: 2D input array.
    :param kernel_x: 1D kernel for the X-axis.
    :param kernel_y: 1D kernel for the Y-axis.
    :param mode: Padding mode - ``"constant"`` (zero) or ``"symmetric"`` (mirror).
    :returns: Convolved array of same shape as input.
    """
    if mode == "constant":
        pad_y, pad_x = 0, 0
        padded = data
    elif mode == "symmetric":
        pad_y = len(kernel_y) // 2
        pad_x = len(kernel_x) // 2
        padded = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode="symmetric")
    else:
        raise ValueError(
            f"Padding mode '{mode}' is not supported. Use 'constant' or 'symmetric'."
        )

    # Convolve: Y-direction then X-direction
    temp = fftconvolve(padded, kernel_y[:, np.newaxis], mode="same")
    result = fftconvolve(temp, kernel_x[np.newaxis, :], mode="same")

    # Crop back to original size if padded
    if pad_y or pad_x:
        result = result[
            pad_y : -pad_y if pad_y else None, pad_x : -pad_x if pad_x else None
        ]

    return result


# ---------------------------------------------------------------------------
# Order-0 filter (unchanged — already optimal)
# ---------------------------------------------------------------------------


def apply_order0_filter(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    mode: str = "constant",
) -> FloatArray2D:
    """
    Perform a 2D weighted moving average (Order-0 Regression) using separable kernels.

    This function treats NaNs in the input data as missing values with zero weight,
    ensuring they do not corrupt the local average.  The result is a
    convolution-based smoothing where each pixel is the weighted mean of its
    neighbours.

    :param data: The 2D input array to be smoothed, potentially containing NaNs.
    :param kernel_x: The 1D X-axis component of the separable smoothing kernel.
    :param kernel_y: The 1D Y-axis component of the separable smoothing kernel.
    :param mode: Padding mode - ``"constant"`` (zero), ``"reflect"``, or ``"symmetric"``.
    :returns: A 2D array of the same shape as ``data`` containing the smoothed values.
    """
    # Assign zero weight to NaNs
    nan_mask = np.isnan(data)
    weights = np.where(nan_mask, 0, 1)
    data_masked = np.where(nan_mask, 0, data)

    # Convolve data and weights
    numerator = convolve_2d_separable(data_masked, kernel_x, kernel_y, mode=mode)
    denominator = convolve_2d_separable(weights, kernel_x, kernel_y, mode=mode)

    # Avoid division by zero and handle edge effects
    return np.where(denominator > 0, numerator / denominator, np.nan)


# ---------------------------------------------------------------------------
# Polynomial filter (orders 1 & 2) — optimised
# ---------------------------------------------------------------------------


def apply_polynomial_filter(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    order: int,
) -> FloatArray2D:
    """
    Apply local polynomial regression filter (orders 1 or 2).

    For each pixel, this fits a polynomial surface to the neighbouring pixels
    using weighted least squares, where the kernel determines the weights.
    The smoothed value is the fitted polynomial evaluated at the centre pixel.

    Order 1 fits a plane (linear):    f(x,y) = c0 + c1·x + c2·y
    Order 2 fits a quadratic surface: f(x,y) = c0 + c1·x + c2·y + c3·x² + c4·xy + c5·y²

    Performance
    -----------
    When ``data`` contains **no NaNs** a fast path is taken (see module
    docstring): the LHS moment matrix is constant over interior pixels, so no
    LHS FFTs are needed and the per-pixel solve is replaced by a single
    ``np.einsum``.  Border pixels (within half-kernel-width of the edge) are
    solved exactly using pre-computed partial kernel moments — no extra FFTs.

    When ``data`` **contains NaNs** the full per-pixel LHS is still required,
    but intermediate arrays are pre-allocated (avoiding costly list-to-array
    conversions) for a meaningful constant-factor speedup.

    :param data: Input 2D array with potential NaNs.
    :param kernel_x: 1D kernel for the X-axis.
    :param kernel_y: 1D kernel for the Y-axis.
    :param order: Polynomial order (1 or 2).
    :returns: Smoothed data array.
    """
    nan_mask = np.isnan(data)
    has_nans = bool(nan_mask.any())

    exponents = _get_polynomial_exponents(order)
    n_params = len(exponents)

    H, W = data.shape
    ky_half = (len(kernel_y) - 1) // 2
    kx_half = (len(kernel_x) - 1) // 2
    y_coords = np.arange(-ky_half, ky_half + 1, dtype=float)
    x_coords = np.arange(-kx_half, kx_half + 1, dtype=float)

    if not has_nans:
        return _apply_polynomial_filter_no_nan(
            data,
            kernel_x,
            kernel_y,
            order,
            exponents,
            n_params,
            y_coords,
            x_coords,
            ky_half,
            kx_half,
        )
    else:
        return _apply_polynomial_filter_nan(
            data,
            nan_mask,
            kernel_x,
            kernel_y,
            exponents,
            n_params,
            y_coords,
            x_coords,
        )


# ---------------------------------------------------------------------------
# Internal: no-NaN fast path
# ---------------------------------------------------------------------------


def _apply_polynomial_filter_no_nan(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    order: int,
    exponents: list[tuple[int, int]],
    n_params: int,
    y_coords: FloatArray1D,
    x_coords: FloatArray1D,
    ky_half: int,
    kx_half: int,
) -> FloatArray2D:
    """
    Fast polynomial filter for data with no NaN values.

    Avoids all LHS FFT convolutions by exploiting the fact that, with
    weights ≡ 1, the moment matrix ``A`` is constant across all interior
    pixels and can be computed analytically from the 1-D kernel moments.
    Border pixels are handled via pre-computed partial moment arrays.
    """
    H, W = data.shape
    max_p = 2 * order

    # ------------------------------------------------------------------
    # 1.  Partial kernel moments (O(max_p × image_size) scalar ops, ~10 ms).
    #     y_partial[p, y]  = Σ_{valid u} u^p · ky[half + u]
    #     where "valid" accounts for zero-padding at the image boundary,
    #     matching fftconvolve's 'same' / constant-padding semantics.
    # ------------------------------------------------------------------
    y_partial = _compute_partial_moments(y_coords, kernel_y, H, ky_half, max_p)
    x_partial = _compute_partial_moments(x_coords, kernel_x, W, kx_half, max_p)

    # ------------------------------------------------------------------
    # 2.  Constant A matrix for interior pixels (H//2, W//2 are interior).
    # ------------------------------------------------------------------
    A_const = np.empty((n_params, n_params))
    for p, (py_p, px_p) in enumerate(exponents):
        for q, (py_q, px_q) in enumerate(exponents):
            A_const[p, q] = (
                y_partial[py_p + py_q, H // 2] * x_partial[px_p + px_q, W // 2]
            )

    # Row 0 of A⁻¹ gives the weights for c₀ (the smoothed value at the centre).
    w0 = np.linalg.inv(A_const)[0]

    # ------------------------------------------------------------------
    # 3.  Build RHS for all polynomial terms (n_params FFT pairs).
    # ------------------------------------------------------------------
    rhs = np.empty((n_params, H, W))
    for i, (py, px) in enumerate(exponents):
        t = fftconvolve(data, (y_coords**py * kernel_y)[:, np.newaxis], mode="same")
        rhs[i] = fftconvolve(t, (x_coords**px * kernel_x)[np.newaxis, :], mode="same")

    # ------------------------------------------------------------------
    # 4.  Interior: c₀ = w0 · rhs  (vectorised over all pixels).
    # ------------------------------------------------------------------
    result = np.einsum("i,i...->...", w0, rhs)

    # ------------------------------------------------------------------
    # 5.  Border correction: exact per-pixel A from partial moments,
    #     then batched linalg.solve on the border pixel subset.
    # ------------------------------------------------------------------
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:ky_half, :] = True
    border_mask[-ky_half:, :] = True
    border_mask[:, :kx_half] = True
    border_mask[:, -kx_half:] = True
    by, bx = np.where(border_mask)

    if len(by) > 0:
        n_border = len(by)
        A_border = np.empty((n_border, n_params, n_params))
        for p, (py_p, px_p) in enumerate(exponents):
            for q, (py_q, px_q) in enumerate(exponents):
                A_border[:, p, q] = (
                    y_partial[py_p + py_q, by] * x_partial[px_p + px_q, bx]
                )
        b_border = np.moveaxis(rhs, 0, -1)[..., np.newaxis][by, bx]
        result[by, bx] = np.linalg.solve(A_border, b_border)[:, 0, 0]

    return result


def _compute_partial_moments(
    coords: FloatArray1D,
    kernel: FloatArray1D,
    size: int,
    half_k: int,
    max_p: int,
) -> FloatArray2D:
    """
    Compute per-row (or per-column) partial kernel moments that reproduce
    what ``fftconvolve(..., mode='same')`` evaluates at each pixel.

    For pixel position ``y``, the valid kernel positions are
    ``j ∈ [y + half_k - min(size-1, y+half_k), y + half_k - max(0, y-half_k)]``
    (clipped to [0, 2·half_k] by the zero-padding of fftconvolve).

    ``partial[p, y] = Σ_{valid j} coords[j]^p · kernel[j]``

    :param coords: Coordinate vector of length ``2·half_k + 1``.
    :param kernel: 1-D kernel of the same length.
    :param size: Image dimension along this axis.
    :param half_k: Half-width of the kernel, ``(len(kernel) - 1) // 2``.
    :param max_p: Highest power needed (``2 · regression_order``).
    :returns: Array of shape ``(max_p + 1, size)``.
    """
    partial = np.zeros((max_p + 1, size))
    # Precompute all powers of the coordinate vector
    coord_pows = np.array(
        [coords**p for p in range(max_p + 1)]
    )  # (max_p+1, 2*half_k+1)

    for y in range(size):
        k_min = max(0, y - half_k)
        k_max = min(size - 1, y + half_k)
        # Kernel indices accessed by fftconvolve 'same' at position y:
        # j = y + half_k - k  for k in [k_min, k_max]
        j_vals = y + half_k - np.arange(k_min, k_max + 1)
        partial[:, y] = coord_pows[:, j_vals] @ kernel[j_vals]

    return partial


# ---------------------------------------------------------------------------
# Internal: NaN path
# ---------------------------------------------------------------------------


def _apply_polynomial_filter_nan(
    data: FloatArray2D,
    nan_mask: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    exponents: list[tuple[int, int]],
    n_params: int,
    y_coords: FloatArray1D,
    x_coords: FloatArray1D,
) -> FloatArray2D:
    """
    Polynomial filter for data containing NaN values.

    The weight function is no longer uniform, so ``A`` varies per pixel and
    the full convolution-based LHS must be computed.  This path matches the
    original algorithm but uses pre-allocated arrays to avoid the overhead of
    ``np.array([generator])`` list construction.
    """
    H, W = data.shape
    weights = np.where(nan_mask, 0.0, 1.0)
    weighted_data = np.where(nan_mask, 0.0, data)

    # ------------------------------------------------------------------
    # 1.  RHS moments: pre-allocate, fill in-place.
    # ------------------------------------------------------------------
    rhs_moments = np.empty((n_params, H, W))
    for i, (py, px) in enumerate(exponents):
        mod_ky = (y_coords**py) * kernel_y
        mod_kx = (x_coords**px) * kernel_x
        t = fftconvolve(weighted_data, mod_ky[:, np.newaxis], mode="same")
        rhs_moments[i] = fftconvolve(t, mod_kx[np.newaxis, :], mode="same")

    # ------------------------------------------------------------------
    # 2.  LHS moments: compute unique power combinations, pre-allocate.
    # ------------------------------------------------------------------
    lhs_matrix = _build_lhs_matrix_prealloc(
        weights, kernel_x, kernel_y, x_coords, y_coords, exponents, n_params
    )

    # ------------------------------------------------------------------
    # 3.  Solve the per-pixel linear system.
    # ------------------------------------------------------------------
    return _solve_pixelwise_regression(lhs_matrix, rhs_moments, data)


def _build_lhs_matrix_prealloc(
    weights: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    x_coords: FloatArray1D,
    y_coords: FloatArray1D,
    exponents: list[tuple[int, int]],
    n_params: int,
) -> FloatArray4D:
    """
    Build the per-pixel LHS matrix ``A`` for the NaN case.

    Identical in semantics to the original ``_build_lhs_matrix`` but uses
    a pre-allocated ``unique_moments`` array instead of
    ``np.array([convolve(...) for ...])`` to avoid Python-level list-to-array
    overhead (~80–90 ms per call for a 512×512 image with a 101-element kernel).
    """
    H, W = weights.shape

    matrix_power_sums = np.array(
        [
            (exponents[p][0] + exponents[q][0], exponents[p][1] + exponents[q][1])
            for p in range(n_params)
            for q in range(n_params)
        ]
    )
    unique_powers, inverse_indices = np.unique(
        matrix_power_sums, axis=0, return_inverse=True
    )
    n_unique = len(unique_powers)

    # Pre-allocate and fill in-place (avoids np.array([generator]) overhead).
    unique_moments = np.empty((n_unique, H, W))
    for i, (py, px) in enumerate(unique_powers):
        t = fftconvolve(weights, (y_coords**py * kernel_y)[:, np.newaxis], mode="same")
        unique_moments[i] = fftconvolve(
            t, (x_coords**px * kernel_x)[np.newaxis, :], mode="same"
        )

    full_moments = unique_moments[inverse_indices]
    return np.moveaxis(full_moments.reshape(n_params, n_params, H, W), [0, 1], [-2, -1])


# ---------------------------------------------------------------------------
# Solver helpers (unchanged)
# ---------------------------------------------------------------------------


def _get_polynomial_exponents(order: int) -> list[tuple[int, int]]:
    """Return list of (power_y, power_x) tuples for polynomial terms."""
    exponents = []
    for py in range(order + 1):
        for px in range(order + 1):
            if py + px <= order:
                exponents.append((py, px))
    return exponents


def _solve_pixelwise_regression(
    lhs_matrix: FloatArray4D,
    rhs_vector: FloatArray3D,
    original_data: FloatArray2D,
) -> FloatArray2D:
    """Solve the linear system for every valid pixel."""
    # rhs_vector shape: (n_params, H, W) -> (H, W, n_params, 1)
    rhs_prepared = np.moveaxis(rhs_vector, 0, -1)[..., np.newaxis]

    result = np.full(original_data.shape, np.nan)

    valid_mask = ~np.isnan(original_data)
    valid_indices = np.where(valid_mask)

    A_valid = lhs_matrix[valid_indices]
    b_valid = rhs_prepared[valid_indices]

    try:
        solutions = np.linalg.solve(A_valid, b_valid)
        result[valid_indices] = solutions[:, 0, 0]
    except np.linalg.LinAlgError:
        _solve_fallback_lstsq(result, lhs_matrix, rhs_prepared, valid_indices)

    return result


def _solve_fallback_lstsq(
    result_array: FloatArray2D,
    lhs: FloatArray4D,
    rhs: FloatArray4D,
    indices: tuple[NDArray[np.intp], ...],
) -> None:
    """Robust fallback solver using Least Squares for difficult pixels."""
    y_idx, x_idx = indices[0], indices[1]
    n_pixels = len(y_idx)

    for i in range(n_pixels):
        y, x = y_idx[i], x_idx[i]
        sol = np.linalg.lstsq(lhs[y, x], rhs[y, x], rcond=None)[0]
        result_array[y, x] = sol[0, 0]


# ---------------------------------------------------------------------------
# Legacy public aliases kept for backward compatibility
# ---------------------------------------------------------------------------


def _build_lhs_matrix(
    weights: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    x_coords: FloatArray1D,
    y_coords: FloatArray1D,
    exponents: list[tuple[int, int]],
) -> FloatArray4D:
    """
    Construct the LHS matrix ``A`` efficiently.

    .. deprecated::
        Prefer ``_build_lhs_matrix_prealloc`` which avoids list-construction
        overhead.  This alias is retained for any callers that reference the
        private function directly.
    """
    return _build_lhs_matrix_prealloc(
        weights, kernel_x, kernel_y, x_coords, y_coords, exponents, len(exponents)
    )
