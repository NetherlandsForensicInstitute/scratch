import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
ALPHA_REGRESSION = 0.7309134280946760


def apply_gaussian_filter(
    data: NDArray[np.floating],
    cutoff_length: float,
    pixel_size: tuple[float, float] = (1.0, 1.0),
    regression_order: int = 0,
    nan_out: bool = True,
    is_high_pass: bool = False,
) -> NDArray[np.floating]:
    """
    Apply a Gaussian filter to 2D data using local polynomial regression (ISO 16610-21).

    This implementation generalizes standard Gaussian filtering to handle missing data (NaNs) using local regression techniques.
    It supports 0th order (weighted average), 1st order (planar fit), and 2nd order (quadratic fit) regression.

    Mathematical Basis:
      - **Order 0**: Equivalent to the Nadaraya-Watson estimator (Kernel Regression).
        It calculates a weighted average where weights are determined by the
        Gaussian kernel and the validity (non-NaN status) of neighboring pixels.

      - **Order 1 & 2**: Local Weighted Least Squares (LOESS).
        It fits a polynomial surface (plane or quadratic) to the local neighborhood
        weighted by the Gaussian kernel. This is effectively a 2D Savitzky-Golay
        filter that is robust to missing data.

    :param data: 2D input array containing float data. May contain NaNs.
    :param cutoff_length: The filter cutoff wavelength in physical units.
    :param pixel_size: Tuple of (y_size, x_size) in physical units.
    :param regression_order: Order of the local polynomial fit (0, 1, or 2).
        0 = Gaussian weighted average.
        1 = Local planar fit (corrects for tilt).
        2 = Local quadratic fit (corrects for curvature).
    :param nan_out: If True, input NaNs remain NaNs in output. If False, the filter attempts to fill gaps based on the local regression.
    :param is_high_pass: If True, returns (input - smoothed). If False, returns smoothed.
    :returns: The filtered 2D array of the same shape as input.
    """
    # 1. Prepare Filter Parameters
    cutoff_pixels = cutoff_length / np.array(pixel_size)
    alpha = ALPHA_REGRESSION if regression_order >= 2 else ALPHA_GAUSSIAN

    # 2. Generate Base 1D Kernels
    kernel_x, kernel_y = _create_normalized_separable_kernels(alpha, cutoff_pixels)

    # 3. Apply Filter Strategy
    if regression_order == 0:
        smoothed = _apply_order0_filter(data, kernel_x, kernel_y)
    else:
        smoothed = _apply_polynomial_filter(data, kernel_x, kernel_y, regression_order)

    # 4. Post-processing
    if nan_out:
        smoothed[np.isnan(data)] = np.nan

    return data - smoothed if is_high_pass else smoothed


def _create_normalized_separable_kernels(
    alpha: float, cutoff_pixels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Create normalized 1D Gaussian kernels for the Y and X axes."""
    # Ensure kernel size is odd and covers sufficient standard deviations
    kernel_dims = 1 + np.ceil(len(cutoff_pixels) * cutoff_pixels).astype(int)
    kernel_dims += 1 - kernel_dims % 2

    kernel_y = _gaussian_1d(kernel_dims[0], cutoff_pixels[0], alpha)
    kernel_x = _gaussian_1d(kernel_dims[1], cutoff_pixels[1], alpha)

    # Normalize so the 2D product sums to 1.0
    total_weight = np.sum(kernel_y) * np.sum(kernel_x)
    return kernel_x / np.sqrt(total_weight), kernel_y / np.sqrt(total_weight)


def _gaussian_1d(size: int, cutoff_pixel: float, alpha: float) -> NDArray[np.floating]:
    """Generate a standard 1D unnormalized Gaussian bell curve."""
    radius = (size - 1) // 2
    # Coordinate vector centered at 0
    coords = np.arange(-radius, radius + 1)

    # Gaussian formula: e^(-pi * (x / (alpha * lambda_c))^2)
    # Note: Division by (alpha * cutoff) scales the Gaussian width
    scale_factor = alpha * cutoff_pixel
    return np.exp(-np.pi * (coords / scale_factor) ** 2) / scale_factor


def _convolve_2d_separable(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Perform fast 2D convolution using separable 1D kernels via FFT."""
    # Convolve columns (Y-direction)
    temp = fftconvolve(data, kernel_y[:, np.newaxis], mode="same")
    # Convolve rows (X-direction)
    return fftconvolve(temp, kernel_x[np.newaxis, :], mode="same")


def _apply_order0_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Apply Order-0 Regression (Weighted Moving Average).

    Equation: Smoothed = (Weights * Data) * K  /  (Weights * K)
    """
    weights = np.where(np.isnan(data), 0.0, 1.0)
    data_filled = np.where(np.isnan(data), 0.0, data)

    numerator = _convolve_2d_separable(data_filled * weights, kernel_x, kernel_y)
    denominator = _convolve_2d_separable(weights, kernel_x, kernel_y)

    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def _apply_polynomial_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    order: int,
) -> NDArray[np.floating]:
    """
    Apply Order-1 or Order-2 Regression (Local Polynomial Fit).

    Solves the linear system A * c = b at each pixel, where:
    - c: vector of polynomial coefficients (c0 is the smoothed value).
    - A: Matrix of weighted coordinate moments.
    - b: Vector of weighted data moments.
    """
    # 1. Setup Coordinate Systems (Normalized to [-1, 1] for stability)
    ny, nx = len(kernel_y), len(kernel_x)
    radius_y, radius_x = (ny - 1) // 2, (nx - 1) // 2

    y_coords = np.arange(-radius_y, radius_y + 1) / (radius_y if ny > 1 else 1.0)
    x_coords = np.arange(-radius_x, radius_x + 1) / (radius_x if nx > 1 else 1.0)

    # 2. Identify Polynomial Terms (Powers of Y and X)
    # We only need terms where power_y + power_x <= order
    exponents = _get_polynomial_exponents(order)
    n_params = len(exponents)

    # 3. Construct the Linear System Components (A matrix and b vector)
    nan_mask = np.isnan(data)
    weights = np.where(nan_mask, 0.0, 1.0)
    weighted_data = np.where(nan_mask, 0.0, data * weights)

    # Calculate RHS vector 'b' (Data Moments)
    # b_k = Convolution(weighted_data, x^px * y^py * Kernel)
    rhs_moments = np.zeros((n_params, *data.shape))
    for i, (py, px) in enumerate(exponents):
        mod_ky = (y_coords**py) * kernel_y
        mod_kx = (x_coords**px) * kernel_x
        rhs_moments[i] = _convolve_2d_separable(weighted_data, mod_kx, mod_ky)

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
    weights: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    x_coords: NDArray[np.floating],
    y_coords: NDArray[np.floating],
    exponents: list[tuple[int, int]],
) -> NDArray[np.floating]:
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
            _convolve_2d_separable(
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
    lhs_matrix: NDArray[np.floating],
    rhs_vector: NDArray[np.floating],
    original_data: NDArray[np.floating],
) -> NDArray[np.floating]:
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
        # Pass the tuple as is, but we ensure the recipient expects a 2D tuple
        _solve_fallback_lstsq(result, lhs_matrix, rhs_prepared, valid_indices)

    return result


def _solve_fallback_lstsq(
    result_array: NDArray[np.floating],
    lhs: NDArray[np.floating],
    rhs: NDArray[np.floating],
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
