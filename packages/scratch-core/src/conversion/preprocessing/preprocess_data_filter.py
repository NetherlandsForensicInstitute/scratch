from math import ceil

import numpy as np
from numpy.typing import NDArray

# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
# alpha = Sqrt((-1 - LambertW(-1, -1 / (2 * exp(1)))) / Pi)
ALPHA_REGRESSION = 0.7309134280946760

# =============================================================================
# 1D Gaussian Filter for Striation-Preserving Processing
# (merged from remove_noise_gaussian.py and remove_shape_gaussian.py)
# =============================================================================


def _cheby_cutoff_to_gauss_sigma(cutoff: float, pixel_size: float) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma using ISO 16610 standard.

    The conversion factor is based on the relationship between Chebyshev
    filter cutoff and equivalent Gaussian standard deviation.

    :param cutoff: Cutoff wavelength in physical units (e.g., meters).
    :param pixel_size: Pixel spacing in the same units as cutoff.
    :return: Gaussian sigma in pixel units.
    """
    # ISO 16610 conversion: sigma = cutoff / (2 * pi * alpha)
    # where alpha = sqrt(ln(2)/pi) for 50% transmission
    cutoff_pixels = cutoff / pixel_size
    return cutoff_pixels * ALPHA_GAUSSIAN


def _apply_nan_weighted_gaussian_1d(
    data: NDArray[np.floating],
    sigma: float,
    radius: int,
) -> NDArray[np.floating]:
    """
    Apply 1D Gaussian filter along rows with NaN-aware weighting.

    NaN values are excluded from the convolution by setting their weight to 0.
    The result is normalized by the sum of weights to compensate for missing data.

    :param data: Input 2D data array (may contain NaN values).
    :param sigma: Standard deviation of Gaussian kernel (in pixels).
    :param radius: Kernel radius in pixels.
    :returns: Filtered data array. NaN positions will have interpolated values
        based on neighboring valid data.
    """
    from scipy import ndimage
    from functools import partial

    # Create 1D Gaussian filter along axis 0 (rows) only
    gaussian_filter = partial(
        ndimage.gaussian_filter,
        sigma=(sigma, 0),  # Filter only along rows
        mode="constant",
        cval=0,
        radius=(radius, 1),
    )

    nan_mask = np.isnan(data)
    # Replace NaN with 0 for filtering
    data_zero_nan = np.where(nan_mask, 0, data)

    # Filter the data
    filtered = gaussian_filter(data_zero_nan)

    # Filter a weight array (1 for valid, 0 for NaN)
    weights = (~nan_mask).astype(float)
    weight_sum = gaussian_filter(weights)

    # Normalize by weights to correct for NaN contributions
    with np.errstate(invalid="ignore", divide="ignore"):
        result = filtered / weight_sum

    return result


def _remove_zero_border(
    data: NDArray[np.floating], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.floating], NDArray[np.bool_], NDArray[np.intp]]:
    """
    Remove zero/invalid borders from masked data.

    Finds the bounding box of valid (masked) data and crops to that region.

    :param data: 2D data array.
    :param mask: Boolean mask (True = valid data).
    :returns: Tuple of (cropped_data, cropped_mask, row_indices).
    """
    # Find rows and columns with any valid data
    valid_rows = np.any(mask, axis=1)
    valid_cols = np.any(mask, axis=0)

    if not np.any(valid_rows) or not np.any(valid_cols):
        # No valid data at all
        return data, mask, np.arange(data.shape[0])

    # Find bounding box
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]

    row_start, row_end = row_indices[0], row_indices[-1] + 1
    col_start, col_end = col_indices[0], col_indices[-1] + 1

    # Crop to bounding box
    cropped_data = data[row_start:row_end, col_start:col_end]
    cropped_mask = mask[row_start:row_end, col_start:col_end]
    range_indices = np.arange(row_start, row_end)

    return cropped_data, cropped_mask, range_indices


def apply_gaussian_filter_1d(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff: float,
    is_high_pass: bool = False,
    cut_borders_after_smoothing: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.intp], NDArray[np.bool_]]:
    """
    Apply 1D Gaussian filter along rows for striation-preserving surface processing.

    This function applies a 1D Gaussian filter only along the first dimension (rows),
    which corresponds to the direction perpendicular to striation marks. This preserves
    striation features while removing either noise (lowpass) or shape (highpass).

    Use Cases:
        - **Lowpass (is_high_pass=False)**: Remove high-frequency noise while preserving
          striation marks. Returns smoothed data.
        - **Highpass (is_high_pass=True)**: Remove large-scale form (curvature, tilt)
          while preserving striation marks. Returns residuals (original - smoothed).

    Algorithm:
        1. Convert cutoff wavelength to Gaussian sigma using ISO standard
        2. Apply 1D Gaussian filter along rows (NaN-aware weighted filtering)
        3. Return smoothed data (lowpass) or residuals (highpass)
        4. Optionally crop border artifacts (sigma pixels from each edge)

    :param depth_data: 2D depth/height data array, or 1D profile. For 2D data, rows should be perpendicular to striation
     direction.
    :param xdim: Pixel spacing in meters (m). Distance between adjacent measurements in the scan.
    cutoff : Cutoff wavelength in meters (m). - For noise removal (lowpass): typical value 250e-6 m (250 µm) - For shape
     removal (highpass): typical value 2000e-6 m (2000 µm)
    :param is_high_pass: If False (default), returns smoothed data (lowpass for noise removal). If True, returns
     residuals (highpass for shape removal).
    :param cut_borders_after_smoothing: If True (default), crop ceil(sigma) pixels from top and bottom edges to remove
     filter artifacts.
    :param mask: Boolean mask array (True = valid data). If provided, masked regions are excluded from filtering. Must
     match depth_data shape.

    :returns filtered_data: Filtered data. For lowpass: smoothed data with noise removed. For highpass: residuals with
     shape removed.
    :returns range_indices: Array of row indices that are valid after cropping. If cut_borders_after_smoothing=True,
     indicates which rows from the original data are included in the output.
    :returns mask: Boolean mask indicating valid data points in the output. True = valid data, False = invalid/masked
     region.

    Notes
    -----
    - Filtering is 1D along rows only (perpendicular to striations)
    - NaN values are handled using weighted filtering
    - Compatible with MATLAB RemoveNoiseGaussian.m and RemoveShapeGaussian.m
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)

    # Check if there are any masked (invalid) regions
    has_masked_regions = np.any(~mask)

    # Calculate Gaussian sigma from cutoff
    sigma = _cheby_cutoff_to_gauss_sigma(cutoff, xdim)
    radius = int(ceil(sigma)) + 1

    # Apply Gaussian lowpass filter
    if has_masked_regions:
        # Use weighted filtering for masked data
        # Set masked regions to NaN for proper handling
        depth_with_nan = depth_data.astype(float).copy()
        depth_with_nan[~mask] = np.nan
        filtered = _apply_nan_weighted_gaussian_1d(depth_with_nan, sigma, radius)
        mask_filtered = mask.copy()
    else:
        # No mask, use standard filtering
        filtered = _apply_nan_weighted_gaussian_1d(
            depth_data.astype(float), sigma, radius
        )
        mask_filtered = mask.copy()

    # Compute output based on filter type
    if is_high_pass:
        # Highpass: return residuals (original - smoothed)
        output = depth_data - filtered
    else:
        # Lowpass: return smoothed data directly
        output = filtered

    # Crop borders if requested
    sigma_int = int(ceil(sigma))

    if cut_borders_after_smoothing:
        if has_masked_regions:
            # Set invalid regions to NaN before cropping
            output[~mask_filtered] = np.nan
            # Remove zero/NaN borders
            cropped_data, cropped_mask, range_indices = _remove_zero_border(
                output, mask_filtered
            )
        else:
            # Simple border crop: remove sigma pixels from top and bottom
            if sigma_int > 0 and depth_data.shape[0] > 2 * sigma_int:
                cropped_data = output[sigma_int:-sigma_int, :]
                cropped_mask = mask_filtered[sigma_int:-sigma_int, :]
                range_indices = np.arange(sigma_int, depth_data.shape[0] - sigma_int)
            else:
                # Data too small to crop
                cropped_data = output
                cropped_mask = mask_filtered
                range_indices = np.arange(depth_data.shape[0])
    else:
        # No cropping
        cropped_data = output
        cropped_mask = mask_filtered
        range_indices = np.arange(depth_data.shape[0])

    return cropped_data, range_indices, cropped_mask
