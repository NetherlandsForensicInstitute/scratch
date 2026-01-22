"""
Unified filtering module for impression and striation preprocessing.

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
from conversion.preprocess_impression.utils import update_mark_data
from conversion.resample import get_scaling_factors

# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
# Approximate value: 0.7309....
ALPHA_REGRESSION = np.sqrt((-1 - lambertw(-1 / (2 * np.exp(1)), -1)) / np.pi).real


def cutoff_to_gaussian_sigma(cutoff: float, pixel_size: float) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma.

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


def gaussian_sigma_to_cutoff(sigma: float, pixel_size: float) -> float:
    """
    Convert Gaussian sigma to cutoff wavelength.

    This is the inverse of cutoff_to_gaussian_sigma. It converts a standard deviation (sigma)
    to the ISO 16610 cutoff wavelength.

    The conversion derives from matching scipy's Gaussian exp(-x²/(2σ²))
    to the ISO Gaussian G(x) = exp(-π(x/(α·λc))²), yielding λc = σ·√(2π)/α.

    :param sigma: Gaussian sigma in pixel units.
    :param pixel_size: Pixel spacing in physical units (e.g., meters).
    :return: Cutoff wavelength in the same units as pixel_size.
    """
    # λc = σ·√(2π)/α where α = √(ln(2)/π) is the ISO 16610 Gaussian constant
    cutoff_pixels = sigma * np.sqrt(2 * np.pi) / ALPHA_GAUSSIAN
    return float(cutoff_pixels * pixel_size)


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


def apply_striation_preserving_filter_1d(
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

    Assumptions:
        - Striations are approximately horizontal (fine alignment corrects small deviations)
        - Striations are elongated horizontal features spanning the image width
        - Striation spatial frequencies fall between the lowpass and highpass cutoffs

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

    # Apply 1D Gaussian filter along y-direction
    cropped_data = _apply_nan_weighted_gaussian_1d(
        scan_image,
        cutoff_length=cutoff,
        mask=mask,
        axis=0,  # Filter along y-direction only
        is_high_pass=is_high_pass,
    )
    cropped_mask = mask

    if cut_borders_after_smoothing:
        # Calculate sigma for border cropping
        sigma = cutoff_to_gaussian_sigma(cutoff, scan_image.scale_x)
        sigma_int = int(ceil(sigma))

        # Check if there are any masked (invalid) regions
        has_masked_regions = np.any(~mask)

        if has_masked_regions:
            cropped_data, cropped_mask = _remove_zero_border(cropped_data, mask)
        elif sigma_int > 0 and scan_image.height > 2 * sigma_int:
            cropped_data = cropped_data[sigma_int:-sigma_int, :]
            cropped_mask = mask[sigma_int:-sigma_int, :]

    return cropped_data, cropped_mask


def apply_gaussian_filter_mark(
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

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=is_high_pass,
        cutoff_length=cutoff,
        regression_order=regression_order,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
    )
    return update_mark_data(mark, filtered_data)


def apply_filter_pipeline(
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
    :param target_scale: Target pixel scale in meters for resampling (None to skip anti-aliasing)
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    :return: Tuple of (filtered mark, anti-aliased-only mark, anti-alias cutoff).
    """
    if target_scale is None:
        mark_anti_aliased, anti_alias_cutoff = mark, None
    else:
        mark_anti_aliased, anti_alias_cutoff = _apply_anti_aliasing(mark, target_scale)

    # Only apply an additional low-pass filter if `lowpass_cutoff` is defined and is bigger than the `anti_alias_cutoff`
    if lowpass_cutoff is not None and (
        anti_alias_cutoff is None or lowpass_cutoff < anti_alias_cutoff
    ):
        mark_filtered = apply_gaussian_filter_mark(
            mark,
            lowpass_cutoff,
            lowpass_regression_order,
            is_high_pass=False,
        )
    else:
        mark_filtered = mark_anti_aliased

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff


def _apply_nan_weighted_gaussian_1d(
    scan_image: ScanImage,
    cutoff_length: float,
    mask: MaskArray | None = None,
    axis: int = 0,
    is_high_pass: bool = False,
) -> NDArray[np.floating]:
    """
    Apply 1D NaN-aware Gaussian filter using FFT convolution.

    This function applies Gaussian filtering along a single axis with NaN handling
    via normalized convolution. Uses the same FFT-based approach as
    apply_gaussian_regression_filter with regression_order=0.

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param cutoff_length: The filter cutoff wavelength in physical units.
    :param mask: Optional boolean mask (True = valid). Combined with scan_image.valid_mask.
    :param axis: Axis to filter along (0=rows/y-direction, 1=columns/x-direction).
    :param is_high_pass: If True, returns (input - smoothed). If False, returns smoothed.
    :returns: The filtered 2D array of the same shape as input.
    """
    pixel_size = scan_image.scale_x if axis == 0 else scan_image.scale_y
    cutoff_pixels = cutoff_length / pixel_size

    # Combine scan_image's valid_mask with external mask
    invalid_mask = ~scan_image.valid_mask
    if mask is not None:
        invalid_mask = invalid_mask | ~mask
    has_nans = np.any(invalid_mask)

    # Prepare data with NaN in invalid positions
    data = scan_image.data.copy()
    if mask is not None:
        data[~mask] = np.nan

    kernel_1d = _create_gaussian_kernel_1d(cutoff_pixels, bool(has_nans))
    kernel_identity = np.array([1.0])
    kernel_x, kernel_y = (
        (kernel_identity, kernel_1d) if axis == 0 else (kernel_1d, kernel_identity)
    )

    smoothed = _apply_order0_filter(
        data, kernel_x, kernel_y, mode="constant" if has_nans else "symmetric"
    )

    # Preserve invalid positions as NaN
    if has_nans:
        smoothed[invalid_mask] = np.nan

    return data - smoothed if is_high_pass else smoothed


def _apply_anti_aliasing(
    mark: Mark,
    target_scale: float,
) -> tuple[Mark, Optional[float]]:
    """
    Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from aliasing when
    resampling to a coarser resolution. Applied when downsampling by >1.5x.

    :param mark: Input mark.
    :param target_scale: Target scale in meters.
    :return: Tuple of (filtered mark, cutoff wavelength applied).
    """
    factors = get_scaling_factors(
        scales=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        target_scale=target_scale,
    )

    # Only filter if downsampling by >1.5x
    if all(r <= 1.5 for r in factors):
        return mark, None

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=False,
        regression_order=0,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        cutoff_length=target_scale,
    )
    return update_mark_data(mark, filtered_data), target_scale


def _create_normalized_separable_kernels(
    alpha: float, cutoff_pixels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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

    kernel_y = _create_normalized_1d_kernel(
        alpha, cutoff_pixels[0], size=kernel_dims[0]
    )
    kernel_x = _create_normalized_1d_kernel(
        alpha, cutoff_pixels[1], size=kernel_dims[1]
    )

    # Each 1D kernel is normalized to sum to 1. Separable convolution uses the outer product
    # of the two kernels (https://en.wikipedia.org/wiki/Outer_product), so the equivalent 2D
    # kernel automatically sums to 1 as well.
    return kernel_x, kernel_y


def _create_gaussian_kernel_1d(
    cutoff_pixels: float,
    has_nans: bool,
) -> NDArray[np.floating]:
    """
    Create a 1D Gaussian kernel with size determined by NaN presence.

    The kernel size calculation differs based on whether the data contains NaNs.
    This is legacy behavior from MATLAB code.

    :param cutoff_pixels: Cutoff wavelength in pixel units.
    :param has_nans: Whether the data contains NaN values.
    :returns: Normalized 1D Gaussian kernel.
    """
    # TODO: Kernel size determination differs for NaN vs non-NaN data (MATLAB legacy).
    # Preference would be to use a single determination, preferably the scipy default.
    sigma = ALPHA_GAUSSIAN * cutoff_pixels / np.sqrt(2 * np.pi)

    if has_nans:
        kernel_size = int(np.ceil(4 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
    else:
        radius = int(np.ceil(3 * sigma))
        kernel_size = 2 * radius + 1

    return _create_normalized_1d_kernel(ALPHA_GAUSSIAN, cutoff_pixels, size=kernel_size)


def _create_normalized_1d_kernel(
    alpha: float,
    cutoff_pixel: float,
    size: int,
) -> NDArray[np.floating]:
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


def _convolve_2d_separable(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    mode: str = "constant",
) -> NDArray[np.floating]:
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
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Remove zero/invalid borders from masked data.

    Finds the bounding box of valid (non-NaN, masked) data and crops to that region.

    :param data: 2D data array (may contain NaN).
    :param mask: Boolean mask (True = valid data).
    :returns: Tuple of (cropped_data, cropped_mask).
    """
    # Consider both mask and NaN values when finding valid region
    valid_data = mask & ~np.isnan(data)

    if not np.any(valid_data):
        # No valid data at all - return empty arrays
        return (
            np.array([]).reshape(0, data.shape[1]),
            np.array([], dtype=bool).reshape(0, data.shape[1]),
        )

    y_slice, x_slice = _determine_bounding_box(valid_data)

    return data[y_slice, x_slice], mask[y_slice, x_slice]
