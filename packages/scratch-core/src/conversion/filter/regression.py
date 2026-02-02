"""
Low-level regression filter implementations.

This module provides kernel creation, convolution operations, and polynomial
regression filters used by the higher-level Gaussian filter functions.
"""

import numpy as np
from scipy.signal import fftconvolve
from numpy.typing import NDArray
from container_models.base import FloatArray1D, FloatArray2D, FloatArray4D, FloatArray3D


def create_normalized_separable_kernels(
    alpha: float, cutoff_pixels: FloatArray1D
) -> tuple[FloatArray1D, FloatArray1D]:
    """
    Create normalized 1D Gaussian kernels for the X and Y axes, where:
      - `kernel_x` is the 1D kernel for the X-axis (row vector).
      - `kernel_y` is the 1D kernel for the Y-axis (column vector).
      - The outer product of these kernels sums to approx 1.0.

    :param alpha: The Gaussian constant (ISO 16610).
    :param cutoff_pixels: Array of [cutoff_y, cutoff_x] in pixel units.
    :returns: A tuple `(kernel_x, kernel_y)`.
    """
    # Ensure kernel size is odd and covers sufficient standard deviations
    kernel_dims = 1 + np.ceil(len(cutoff_pixels) * cutoff_pixels).astype(int)
    kernel_dims += 1 - kernel_dims % 2

    kernel_y = create_normalized_1d_kernel(alpha, cutoff_pixels[0], size=kernel_dims[0])
    kernel_x = create_normalized_1d_kernel(alpha, cutoff_pixels[1], size=kernel_dims[1])

    # Each 1D kernel is normalized to sum to 1. Separable convolution uses the outer product
    # of the two kernels (https://en.wikipedia.org/wiki/Outer_product), so the equivalent 2D
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
    :param mode: Padding mode - "constant" (zero) or "symmetric" (mirror).
    :returns: Convolved array of same shape as input.
    """
    # Prepare: apply explicit padding for symmetric mode
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


def apply_order0_filter(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    mode: str = "constant",
) -> FloatArray2D:
    """
    Perform a 2D weighted moving average (Order-0 Regression) using separable kernels.

    This function treats NaNs in the input data as missing values with zero weight,
    ensuring they do not corrupt the local average. The result is a convolution-based
    smoothing where each pixel is the weighted mean of its neighbors.

    :param data: The 2D input array to be smoothed, potentially containing NaNs.
    :param kernel_x: The 1D X-axis component of the separable smoothing kernel.
    :param kernel_y: The 1D Y-axis component of the separable smoothing kernel.
    :param mode: Padding mode - "constant" (zero), "reflect", or "symmetric".
    :returns: A 2D array of the same shape as `data` containing the smoothed values.
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


def apply_polynomial_filter(
    data: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    order: int,
) -> FloatArray2D:
    """
    Apply local polynomial regression filter (orders 1 or 2).

    For each pixel, this fits a polynomial surface to the neighboring pixels using
    weighted least squares, where the kernel determines the weights. The smoothed
    value is the fitted polynomial evaluated at the center pixel.

    Order 1 fits a plane (linear): f(x,y) = c0 + c1*x + c2*y
    Order 2 fits a quadratic surface: f(x,y) = c0 + c1*x + c2*y + c3*x² + c4*xy + c5*y²

    :param data: Input 2D array with potential NaNs.
    :param kernel_x: 1D kernel for the X-axis.
    :param kernel_y: 1D kernel for the Y-axis.
    :param order: Polynomial order (1 or 2).
    :returns: Smoothed data array.
    """
    # 1. Prepare Data
    nan_mask = np.isnan(data)
    weights = np.where(nan_mask, 0, 1)
    weighted_data = np.where(nan_mask, 0, data) * weights

    # 2. Generate Coordinate Grids
    # These represent (x - x') and (y - y') for all kernel positions
    ky_len, kx_len = len(kernel_y), len(kernel_x)
    y_coords = np.arange(-(ky_len // 2), ky_len // 2 + 1).astype(float)
    x_coords = np.arange(-(kx_len // 2), kx_len // 2 + 1).astype(float)

    # 3. Build Right-Hand Side (RHS): Weighted Moments
    # b_j = Convolution(data * weights, x^px * y^py * Kernel)
    exponents = _get_polynomial_exponents(order)
    n_params = len(exponents)
    rhs_moments = np.zeros((n_params, *data.shape))

    for i, (py, px) in enumerate(exponents):
        mod_ky = (y_coords**py) * kernel_y
        mod_kx = (x_coords**px) * kernel_x
        rhs_moments[i] = convolve_2d_separable(weighted_data, mod_kx, mod_ky)

    # Calculate LHS Matrix 'A' (Weight Moments)
    # A_jk = Convolution(weights, x^(px_j + px_k) * y^(py_j + py_k) * Kernel)
    lhs_matrix = _build_lhs_matrix(
        weights, kernel_x, kernel_y, x_coords, y_coords, exponents
    )

    # 4. Solve the System (A * c = b) per pixel
    return _solve_pixelwise_regression(lhs_matrix, rhs_moments, data)


def _get_polynomial_exponents(order: int) -> list[tuple[int, int]]:
    """Return list of (power_y, power_x) tuples for polynomial terms."""
    exponents = []
    for py in range(order + 1):
        for px in range(order + 1):
            if py + px <= order:
                exponents.append((py, px))
    return exponents


def _build_lhs_matrix(
    weights: FloatArray2D,
    kernel_x: FloatArray1D,
    kernel_y: FloatArray1D,
    x_coords: FloatArray1D,
    y_coords: FloatArray1D,
    exponents: list[tuple[int, int]],
) -> FloatArray4D:
    """
    Construct the LHS matrix 'A' efficiently.

    Optimization: Many terms in the matrix are identical (e.g., x*y appears multiple times).
    We compute unique moment sums first, then map them to the matrix structure.
    """
    n_params = len(exponents)

    # Calculate sum of powers for every cell in the matrix (A_pq = term_p * term_q)
    matrix_power_sums = np.array(
        [
            (exponents[p][0] + exponents[q][0], exponents[p][1] + exponents[q][1])
            for p in range(n_params)
            for q in range(n_params)
        ]
    )

    # Find unique combinations of powers to avoid redundant convolutions
    unique_powers, inverse_indices = np.unique(
        matrix_power_sums, axis=0, return_inverse=True
    )

    # Compute convolutions for unique powers
    unique_moments = np.array(
        [
            convolve_2d_separable(
                weights, (x_coords**px) * kernel_x, (y_coords**py) * kernel_y
            )
            for py, px in unique_powers
        ]
    )

    # Reconstruct the full (n_params * n_params) matrix for every pixel
    # Current shape: (unique_terms, H, W) -> Map to -> (n_params^2, H, W)
    full_moments = unique_moments[inverse_indices]

    # Reshape to (H, W, n_params, n_params)
    # We move axes so H, W are first, creating a matrix for every pixel
    return np.moveaxis(
        full_moments.reshape(n_params, n_params, *weights.shape), [0, 1], [-2, -1]
    )


def _solve_pixelwise_regression(
    lhs_matrix: FloatArray4D,
    rhs_vector: FloatArray3D,
    original_data: FloatArray2D,
) -> FloatArray2D:
    """Solve the linear system for every valid pixel."""
    # rhs_vector shape: (n_params, H, W) -> (H, W, n_params, 1)
    rhs_prepared = np.moveaxis(rhs_vector, 0, -1)[..., np.newaxis]

    # Output array
    result = np.full(original_data.shape, np.nan)

    # Identify pixels where input data exists (standard behavior is to skip Input-NaNs)
    valid_mask = ~np.isnan(original_data)
    valid_indices = np.where(valid_mask)

    A_valid = lhs_matrix[valid_indices]
    b_valid = rhs_prepared[valid_indices]

    try:
        # Batch solve is much faster
        solutions = np.linalg.solve(A_valid, b_valid)
        result[valid_indices] = solutions[:, 0, 0]  # c0 is the smoothed value
    except np.linalg.LinAlgError:
        # Pass the tuple as is, but we ensure the recipient expects a 2D tuple.
        # This modifies `result` in place.
        _solve_fallback_lstsq(result, lhs_matrix, rhs_prepared, valid_indices)

    return result


def _solve_fallback_lstsq(
    result_array: FloatArray2D,
    lhs: FloatArray4D,
    rhs: FloatArray4D,
    indices: tuple[
        NDArray[np.intp], ...
    ],  # Use ellipsis to allow variadic tuples of index arrays
) -> None:
    """Robust fallback solver using Least Squares for difficult pixels."""
    # We explicitly extract y and x to make the 2D logic clear to the reader
    y_idx, x_idx = indices[0], indices[1]
    n_pixels = len(y_idx)

    for i in range(n_pixels):
        y, x = y_idx[i], x_idx[i]
        # lstsq returns (solution, residuals, rank, singular_values)
        sol = np.linalg.lstsq(lhs[y, x], rhs[y, x], rcond=None)[0]
        result_array[y, x] = sol[0, 0]
