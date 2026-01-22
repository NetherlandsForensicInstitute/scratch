"""
Core Gaussian filtering functions for surface texture analysis.

This module provides Gaussian regression filtering following ISO 16610 standards,
including NaN-aware filtering and striation-preserving 1D filters.
"""

from math import ceil
from typing import Optional
from scipy.special import lambertw

import numpy as np
from numpy.typing import NDArray

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.filter.regression import (
    create_normalized_separable_kernels,
    create_gaussian_kernel_1d,
    apply_order0_filter,
    apply_polynomial_filter,
)
from conversion.filter.utils import remove_zero_border

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
    kernel_x, kernel_y = create_normalized_separable_kernels(alpha, cutoff_pixels)

    # 3. Apply Filter Strategy
    if regression_order == 0:
        smoothed = apply_order0_filter(data, kernel_x, kernel_y)
    else:
        smoothed = apply_polynomial_filter(data, kernel_x, kernel_y, regression_order)

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
            cropped_data, cropped_mask = remove_zero_border(cropped_data, mask)
        elif sigma_int > 0 and scan_image.height > 2 * sigma_int:
            cropped_data = cropped_data[sigma_int:-sigma_int, :]
            cropped_mask = mask[sigma_int:-sigma_int, :]

    return cropped_data, cropped_mask


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

    kernel_1d = create_gaussian_kernel_1d(cutoff_pixels, bool(has_nans), ALPHA_GAUSSIAN)
    kernel_identity = np.array([1.0])
    kernel_x, kernel_y = (
        (kernel_identity, kernel_1d) if axis == 0 else (kernel_1d, kernel_identity)
    )

    smoothed = apply_order0_filter(
        data, kernel_x, kernel_y, mode="constant" if has_nans else "symmetric"
    )

    # Preserve invalid positions as NaN
    if has_nans:
        smoothed[invalid_mask] = np.nan

    return data - smoothed if is_high_pass else smoothed
