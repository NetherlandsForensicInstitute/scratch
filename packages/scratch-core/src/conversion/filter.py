"""Unified filtering module for impression and striation preprocessing.

This module provides Gaussian regression filtering and related operations for
surface texture analysis, following ISO 16610 standards.
"""

from math import ceil
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve
from scipy.special import lambertw

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.mask import _determine_bounding_box
from conversion.resample import get_scaling_factors

# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
ALPHA_REGRESSION = np.sqrt(
    (-1 - lambertw(-1 / (2 * np.exp(1)), -1)) / np.pi
).real  # 0.7309...


def cutoff_to_gaussian_sigma(cutoff: float, pixel_size: float) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma for use with scipy's gaussian_filter.

    This function converts the ISO 16610 cutoff wavelength to a standard deviation (sigma)
    compatible with scipy.ndimage.gaussian_filter, which uses exp(-x²/(2σ²)).

    The conversion derives from matching the ISO Gaussian G(x) = exp(-π(x/(α·λc))²)
    to scipy's Gaussian exp(-x²/(2σ²)), yielding σ = α·λc/√(2π).

    :param cutoff: Cutoff wavelength in physical units (e.g., meters).
    :param pixel_size: Pixel spacing in the same units as cutoff.
    :return: Gaussian sigma in pixel units.
    """
    cutoff_pixels = cutoff / pixel_size
    # σ = α·λc/√(2π) where α = √(ln(2)/π) is the ISO 16610 Gaussian constant
    return float(ALPHA_GAUSSIAN * cutoff_pixels / np.sqrt(2 * np.pi))


def apply_gaussian_regression_filter(
    data: NDArray[np.floating],
    cutoff_length: float,
    pixel_size: tuple[float, float] = (1.0, 1.0),
    regression_order: int = 0,
    nan_out: bool = True,
    is_high_pass: bool = False,
) -> NDArray[np.floating]:
    """
    Apply a 2D Savitzky-Golay filter with Gaussian weighting via local polynomial regression (ISO 16610-21).

    This implementation generalizes standard Gaussian filtering to handle missing data (NaNs) using local
    regression techniques. It supports 0th order (Gaussian Kernel weighted average), 1st order (planar fit),
    and 2nd order (quadratic fit) regression.

    Explanation of Regression Orders:
      - **Order 0**: Equivalent to the Nadaraya-Watson estimator. It calculates a weighted average where weights
        are determined by the Gaussian kernel and the validity (non-NaN status) of neighboring pixels.
      - **Order 1 & 2**: Local Weighted Least Squares (LOESS). It fits a polynomial surface (plane or quadratic) to
        the local neighborhood weighted by the Gaussian kernel. This acts as a robust 2D Savitzky-Golay filter.

    Mathematical basis:
      - Approximate a signal s(x, y) from noisy data f(x, y) = s(x, y) + e(x, y) using weighted local regression.
      - The approximation b(x, y) is calculated as the fitted value at point (x, y) using a weighted least squares
        approach. Weights are non-zero within the neighborhood [x - rx, x + rx] and [y - ry, y + ry], following a
        Gaussian distribution with standard deviations proportional to rx and ry.
      - Optimization:
        For **Order 0**, the operation is mathematically equivalent to a normalized convolution. This implementation
        uses FFT-based convolution for performance gains compared to pixel-wise regression.

    :param data: 2D input array containing float data. May contain NaNs.
    :param cutoff_length: The filter cutoff wavelength in physical units.
    :param pixel_size: Tuple of (y_size, x_size) in physical units.
    :param regression_order: Order of the local polynomial fit (0, 1, or 2).
        0 = Gaussian weighted average.
        1 = Local planar fit (corrects for tilt).
        2 = Local quadratic fit (corrects for quadratic curvature).
    :param nan_out: If True, input NaNs remain NaNs in output. If False, the filter attempts to
        fill gaps based on the local regression.
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


def apply_gaussian_filter_1d_to_scan_image(
    scan_image: ScanImage,
    cutoff: float,
    is_high_pass: bool = False,
    cut_borders_after_smoothing: bool = True,
    mask: Optional[MaskArray] = None,
) -> tuple[NDArray[np.floating], MaskArray]:
    """
    Apply 1D Gaussian filter along rows (y-direction) for striation-preserving surface processing.

    This function applies a 1D Gaussian filter only along axis 0 (rows/y-direction),
    which smooths vertically while preserving horizontal striation features.

    Use Cases:
        - **Lowpass (is_high_pass=False)**: Remove high-frequency noise while preserving
          striation marks. Returns smoothed data.
        - **Highpass (is_high_pass=True)**: Remove large-scale form (curvature, tilt)
          while preserving striation marks. Returns residuals (original - smoothed).

    Algorithm:
        1. Convert cutoff wavelength to Gaussian sigma using ISO standard
        2. Apply 1D Gaussian filter along rows (NaN-aware weighted filtering)
        3. Return smoothed data (lowpass) or residuals (highpass)
        4. Optionally crop border artifacts (sigma pixels from top and bottom edges)

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param cutoff: Cutoff wavelength in meters (m).
    :param is_high_pass: If False, returns smoothed data (lowpass). If True, returns residuals (highpass).
    :param cut_borders_after_smoothing: If True, crop ceil(sigma) pixels from top and bottom edges.
    :param mask: Boolean mask array (True = valid data). Must match depth_data shape.

    :returns filtered_data: Filtered data.
    :returns mask: Boolean mask indicating valid data points in the output.
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(scan_image.data.shape, dtype=bool)

    # Apply 1D Gaussian filter along y-direction using shared implementation
    depth_with_nan = scan_image.data.copy()
    depth_with_nan[~mask] = np.nan
    output = _apply_nan_weighted_gaussian_1d(
        data=depth_with_nan,
        cutoff_length=cutoff,
        pixel_size=scan_image.scale_x,
        axis=0,  # Filter along y-direction only
        is_high_pass=is_high_pass,
    )

    # Check if there are any masked (invalid) regions
    has_masked_regions = np.any(~mask)

    cropped_data = output
    cropped_mask = mask

    if cut_borders_after_smoothing:
        # Calculate sigma for border cropping
        sigma = cutoff_to_gaussian_sigma(cutoff, scan_image.scale_x)
        sigma_int = int(ceil(sigma))

        if has_masked_regions:
            output_with_nan = output.copy()
            output_with_nan[~mask] = np.nan
            cropped_data, cropped_mask, _ = _remove_zero_border(output_with_nan, mask)
        elif sigma_int > 0 and scan_image.height > 2 * sigma_int:
            cropped_data = output[sigma_int:-sigma_int, :]
            cropped_mask = mask[sigma_int:-sigma_int, :]

    return cropped_data, cropped_mask


def apply_gaussian_filter_to_mark(
    mark: Mark,
    cutoff: float,
    regression_order: int,
    is_high_pass: bool,
) -> Mark:
    """
    Apply 2D Gaussian filter to mark data.

    :param mark: Input mark.
    :param cutoff: Filter cutoff length.
    :param regression_order: Order of the local polynomial fit (0, 1, or 2).
    :param is_high_pass: If True, apply high-pass filter; otherwise low-pass.
    :return: Filtered mark.
    """
    from conversion.preprocess_impression.utils import update_mark_data

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=is_high_pass,
        cutoff_length=cutoff,
        regression_order=regression_order,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
    )
    return update_mark_data(mark, filtered_data)


def apply_filtering_pipeline(
    mark: Mark,
    target_scale: Optional[float],
    lowpass_cutoff: Optional[float],
    lowpass_regression_order: int,
) -> tuple[Mark, Mark, Optional[float]]:
    """
    Apply the filtering pipeline to a leveled mark: anti-aliasing and low-pass filtering.

    Anti-aliasing is implemented using a zero-order Gaussian regression filter, which effectively
    acts as a low-pass filter to suppress frequencies above the Nyquist limit when resampling.

    :param mark: Leveled mark.
    :param target_scale: Target pixel scale in meters for resampling
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    :return: Tuple of (filtered mark, anti-aliased-only mark, anti-alias cutoff).
    """
    mark_anti_aliased, anti_alias_cutoff = _apply_anti_aliasing(mark, target_scale)

    mark_filtered = mark_anti_aliased

    # Only apply an additional low-pass filter if `lowpass_cutoff` is defined and is bigger than the `anti_alias_cutoff`
    if lowpass_cutoff is not None and (
        anti_alias_cutoff is None or lowpass_cutoff < anti_alias_cutoff
    ):
        mark_filtered = apply_gaussian_filter_to_mark(
            mark,
            lowpass_cutoff,
            lowpass_regression_order,
            is_high_pass=False,
        )

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff


def _apply_nan_weighted_gaussian_1d(
    data: NDArray[np.floating],
    cutoff_length: float,
    pixel_size: float,
    axis: int = 0,
    is_high_pass: bool = False,
) -> NDArray[np.floating]:
    """
    Apply 1D NaN-aware Gaussian filter using FFT convolution.

    This function applies Gaussian filtering along a single axis with NaN handling
    via normalized convolution. Uses the same FFT-based approach as
    apply_gaussian_regression_filter with regression_order=0.

    Matches MATLAB SmoothMod.m behavior:
      - Data with NaNs: Uses zero padding + normalized convolution (like NanConv with 'edge')
      - Data without NaNs: Uses symmetric padding (like DIPimage smooth())

    :param data: 2D input array containing float data. May contain NaNs.
    :param cutoff_length: The filter cutoff wavelength in physical units.
    :param pixel_size: Pixel spacing in the same units as cutoff_length.
    :param axis: Axis to filter along (0=rows/y-direction, 1=columns/x-direction).
    :param is_high_pass: If True, returns (input - smoothed). If False, returns smoothed.
    :returns: The filtered 2D array of the same shape as input.
    """
    cutoff_pixels = cutoff_length / pixel_size

    # Create 1D kernel for the specified axis
    kernel_1d = _create_normalized_1d_kernel(ALPHA_GAUSSIAN, cutoff_pixels)

    # Set up separable kernels based on axis
    if axis == 0:
        kernel_y = kernel_1d
        kernel_x = np.array([1.0])
    else:
        kernel_y = np.array([1.0])
        kernel_x = kernel_1d

    # Match MATLAB SmoothMod.m: different boundary modes based on NaN presence
    # - NaNs present: zero padding + edge correction via normalization (NanConv path)
    # - No NaNs: symmetric padding (DIPimage smooth() path)
    has_nans = np.any(np.isnan(data))
    mode = "constant" if has_nans else "symmetric"

    smoothed = _apply_order0_filter(data, kernel_x, kernel_y, mode=mode)
    return data - smoothed if is_high_pass else smoothed


def _apply_anti_aliasing(
    mark: Mark,
    target_scale: Optional[float],
) -> tuple[Mark, Optional[float]]:
    """
    Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from aliasing when
    resampling to a coarser resolution. Applied when downsampling by >1.5x.

    :param mark: Input mark.
    :param target_scale: Target scale in meters.
    :return: Tuple of (filtered mark, cutoff wavelength applied).
    """
    from conversion.preprocess_impression.utils import update_mark_data

    if target_scale is None:
        return mark, None

    factors = get_scaling_factors(
        scales=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        target_scale=target_scale,
    )

    # Only filter if downsampling by >1.5x
    if all(r <= 1.5 for r in factors):
        return mark, None

    cutoff = target_scale

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=False,
        regression_order=0,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        cutoff_length=cutoff,
    )
    return update_mark_data(mark, filtered_data), cutoff


def _create_normalized_separable_kernels(
    alpha: float, cutoff_pixels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Create normalized 1D Gaussian kernels for the X and Y axes, where:
      - `kernel_x` is the 1D horizontal kernel (row vector).
      - `kernel_y` is the 1D vertical kernel (column vector).
      - The outer product of these kernels sums to approx 1.0.

    :param alpha: The Gaussian constant (ISO 16610).
    :param cutoff_pixels: Array of [cutoff_y, cutoff_x] in pixel units.
    :returns: A tuple `(kernel_x, kernel_y)`.
    """
    # Ensure kernel size is odd and covers sufficient standard deviations
    kernel_dims = 1 + np.ceil(len(cutoff_pixels) * cutoff_pixels).astype(int)
    kernel_dims += 1 - kernel_dims % 2

    kernel_y = _gaussian_1d(kernel_dims[0], cutoff_pixels[0], alpha)
    kernel_x = _gaussian_1d(kernel_dims[1], cutoff_pixels[1], alpha)

    # Normalize so the 2D product sums to 1.0
    total_weight = np.sum(kernel_y) * np.sum(kernel_x)
    return kernel_x / np.sqrt(total_weight), kernel_y / np.sqrt(total_weight)


def _gaussian_1d(size: int, cutoff_pixel: float, alpha: float) -> NDArray[np.floating]:
    """
    Generate a 1D Gaussian curve scaled by the cutoff wavelength.

    :param size: The length of the kernel (must be odd).
    :param cutoff_pixel: The cutoff wavelength in pixel units.
    :param alpha: The Gaussian constant.
    :returns: 1D array of Gaussian weights.
    """
    radius = (size - 1) // 2
    # Coordinate vector centered at 0
    coords = np.arange(-radius, radius + 1)

    # Gaussian formula: e^(-pi * (x / (alpha * lambda_c))^2)
    # Note: Division by (alpha * cutoff) scales the Gaussian width
    scale_factor = alpha * cutoff_pixel
    return np.exp(-np.pi * (coords / scale_factor) ** 2) / scale_factor


def _create_normalized_1d_kernel(
    alpha: float, cutoff_pixel: float
) -> NDArray[np.floating]:
    """
    Create a normalized 1D Gaussian kernel matching scipy.ndimage behavior.

    :param alpha: The Gaussian constant (ISO 16610).
    :param cutoff_pixel: Cutoff wavelength in pixel units.
    :returns: Normalized 1D Gaussian kernel that sums to 1.
    """
    # Convert cutoff to sigma (same formula as cutoff_to_gaussian_sigma)
    sigma = alpha * cutoff_pixel / np.sqrt(2 * np.pi)

    # Match scipy.ndimage truncation: kernel covers 4 standard deviations
    truncate = 4.0
    radius = int(truncate * sigma + 0.5)

    # Create scipy-style Gaussian: exp(-x²/(2σ²))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)

    return kernel / np.sum(kernel)  # Normalize to sum to 1


def _convolve_2d_separable(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    mode: str = "constant",
) -> NDArray[np.floating]:
    """
    Perform fast 2D convolution using separable 1D kernels via FFT.

    :param data: 2D input array.
    :param kernel_x: 1D horizontal kernel.
    :param kernel_y: 1D vertical kernel.
    :param mode: Padding mode - "constant" (zero), "reflect", or "symmetric".
    :returns: Convolved array of same shape as input.
    """
    if mode == "constant":
        # Original behavior - zero padding (implicit in fftconvolve)
        temp = fftconvolve(data, kernel_y[:, np.newaxis], mode="same")
        return fftconvolve(temp, kernel_x[np.newaxis, :], mode="same")
    elif mode == "symmetric":
        # For symmetric modes, pad data explicitly
        pad_y = len(kernel_y) // 2
        pad_x = len(kernel_x) // 2

        padded = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode=mode)

        # Convolve columns (Y-direction)
        temp = fftconvolve(padded, kernel_y[:, np.newaxis], mode="same")
        # Convolve rows (X-direction)
        result = fftconvolve(temp, kernel_x[np.newaxis, :], mode="same")

        # Crop back to original size
        return result[
            pad_y : -pad_y if pad_y else None, pad_x : -pad_x if pad_x else None
        ]
    else:
        raise ValueError("This padding mode is not implemented")


def _apply_order0_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    mode: str = "constant",
) -> NDArray[np.floating]:
    """
    Perform a 2D weighted moving average (Order-0 Regression) using separable kernels.

    This function treats NaNs in the input data as missing values with zero weight,
    ensuring they do not corrupt the local average. The result is a convolution-based
    smoothing where each pixel is the weighted mean of its neighbors.

    :param data: The 2D input array to be smoothed, potentially containing NaNs.
    :param kernel_x: The 1D horizontal component of the separable smoothing kernel.
    :param kernel_y: The 1D vertical component of the separable smoothing kernel.
    :param mode: Padding mode - "constant" (zero), "reflect", or "symmetric".
    :returns: A 2D array of the same shape as `data` containing the smoothed values.
    """
    # Assign zero weight to NaNs
    weights = np.where(np.isnan(data), 0, 1)
    data_masked = np.where(np.isnan(data), 0, data)

    # Convolve data and weights
    numerator = _convolve_2d_separable(data_masked, kernel_x, kernel_y, mode=mode)
    denominator = _convolve_2d_separable(weights, kernel_x, kernel_y, mode=mode)

    # Avoid division by zero and handle edge effects
    return np.where(denominator > 0, numerator / denominator, np.nan)


def _apply_polynomial_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    order: int,
) -> NDArray[np.floating]:
    """
    Apply local polynomial regression filter (orders 1 or 2).

    This function solves the weighted least squares problem for each pixel:
        min_c Σ w(x-x', y-y') · (f(x', y') - Σ c_k · basis_k(x'-x, y'-y))²

    :param data: Input 2D array with potential NaNs.
    :param kernel_x: 1D horizontal kernel.
    :param kernel_y: 1D vertical kernel.
    :param order: Polynomial order (1 or 2).
    :returns: Smoothed data array.
    """
    # 1. Prepare Data
    weights = np.where(np.isnan(data), 0, 1)
    weighted_data = np.where(np.isnan(data), 0, data) * weights

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
        # Pass the tuple as is, but we ensure the recipient expects a 2D tuple.
        # This modifies `result` in place.
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


def _remove_zero_border(
    data: NDArray[np.floating], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.intp]]:
    """
    Remove zero/invalid borders from masked data.

    Finds the bounding box of valid (non-NaN, masked) data and crops to that region.

    :param data: 2D data array (may contain NaN).
    :param mask: Boolean mask (True = valid data).
    :returns: Tuple of (cropped_data, cropped_mask, row_indices of the bounding box).
    """
    # Consider both mask and NaN values when finding valid region
    valid_data = mask & ~np.isnan(data)

    # Find rows and columns with any valid data
    valid_rows = np.any(valid_data, axis=1)
    valid_cols = np.any(valid_data, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        # No valid data at all - return empty arrays
        return (
            np.array([]).reshape(0, data.shape[1]),
            np.array([], dtype=bool).reshape(0, data.shape[1]),
            np.array([], dtype=np.intp),
        )

    # Find bounding box and crop
    # _determine_bounding_box returns (y_slice, x_slice) i.e. (row_slice, col_slice)
    y_slice, x_slice = _determine_bounding_box(valid_data)
    cropped_data = data[y_slice, x_slice]
    cropped_mask = mask[y_slice, x_slice]
    range_indices = np.arange(y_slice.start, y_slice.stop)

    return cropped_data, cropped_mask, range_indices
