"""Remove high-frequency noise from surface data using Gaussian lowpass filter.

This module provides functionality to remove high-frequency noise (such as
scanner noise or fine scratches) from surface scan data, preserving the
larger-scale features like striation marks.

Unlike shape removal (highpass), noise removal uses a lowpass filter that
returns the smoothed data directly (not residuals).

Migrated from MATLAB: RemoveNoiseGaussian.m
"""

from dataclasses import dataclass
from functools import partial
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from conversion.cheby_cutoff_to_gauss_sigma import cheby_cutoff_to_gauss_sigma


@dataclass
class NoiseRemovalResult:
    """Result container for noise removal operation.

    Attributes:
        depth_data: The filtered depth data with noise removed (smoothed).
            This contains low-frequency content (striation marks without noise).
        range_indices: Array of row indices that are valid after cropping.
            If cut_borders_after_smoothing=True, this indicates which rows
            from the original data are included in the output.
        mask: Boolean mask indicating valid data points in the output.
            True = valid data, False = invalid/masked region.
    """

    depth_data: NDArray[np.floating]
    range_indices: NDArray[np.intp]
    mask: NDArray[np.bool_]


def _apply_nan_weighted_gaussian(
    data: NDArray[np.floating],
    sigma: tuple[float, float],
    radius: tuple[int, int],
) -> NDArray[np.floating]:
    """
    Apply Gaussian filter with NaN-aware weighting.

    NaN values are excluded from the convolution by setting their weight to 0.
    The result is normalized by the sum of weights to compensate for missing data.

    :param data: Input 2D data array (may contain NaN values).
    :param sigma: Standard deviation of Gaussian kernel for each axis (row, col).
    :param radius: Kernel radius for each axis.
    :return: Filtered data array. NaN positions will have interpolated values
        based on neighboring valid data.
    """
    gaussian_filter = partial(
        ndimage.gaussian_filter,
        sigma=sigma,
        mode="constant",
        cval=0,
        radius=radius,
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
    :return: Tuple of (cropped_data, cropped_mask, row_indices).
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


def remove_noise_gaussian(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_lo: float = 250.0,
    cut_borders_after_smoothing: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> NoiseRemovalResult:
    """
    Remove noise from surface data using Gaussian lowpass filter.

    This function removes high-frequency noise from surface scan data using
    a Gaussian lowpass filter. The result is smoothed data that preserves
    larger-scale features like striation marks while removing fine noise.

    Unlike shape removal (which returns residuals), noise removal returns
    the smoothed data directly.

    The filtering is applied only along the first dimension (rows), which
    corresponds to the direction perpendicular to striation marks.

    Algorithm:
        1. Convert cutoff wavelength to Gaussian sigma using ISO standard
        2. Apply 1D Gaussian lowpass filter along rows
        3. Return smoothed data (not residuals)
        4. Optionally crop border artifacts (sigma pixels from each edge)

    :param depth_data: 2D depth/height data array, or 1D profile. For 2D data,
        rows should be perpendicular to striation direction.
    :param xdim: Pixel spacing in meters (m). Distance between adjacent
        measurements in the scan.
    :param cutoff_lo: Low-frequency cutoff wavelength in micrometers (um).
        Smaller values remove more high-frequency noise. Default 250 um is
        typical for noise removal while preserving striation features.
    :param cut_borders_after_smoothing: If True, crop ceil(sigma) pixels
        from top and bottom edges to remove filter artifacts. Default True.
    :param mask: Optional boolean mask array (True = valid data). If provided,
        masked regions are excluded from filtering. Must match depth_data shape.
    :return: NoiseRemovalResult containing:
        - depth_data: Smoothed data with noise removed
        - range_indices: Valid row indices after cropping
        - mask: Output mask after processing

    Example:
        >>> import numpy as np
        >>> # Create synthetic surface with signal + noise
        >>> x = np.linspace(0, 100, 500)
        >>> signal = np.sin(2 * np.pi * x / 20)  # Striation pattern
        >>> noise = np.random.randn(500) * 0.5  # High-frequency noise
        >>> surface = np.tile(signal + noise, (100, 1)).T
        >>> # Remove noise
        >>> result = remove_noise_gaussian(surface, xdim=1e-6, cutoff_lo=250)
        >>> # result.depth_data now contains mostly signal, noise removed

    Note:
        - For 2D data, filtering is 1D along rows only (sigma_col = 0)
        - NaN values are handled using weighted filtering
        - Returns smoothed data (lowpass), not residuals (unlike shape removal)
        - The function matches MATLAB RemoveNoiseGaussian.m behavior
    """
    # Ensure depth_data is 2D
    depth_data = np.atleast_2d(depth_data)

    # If input was 1D row vector, transpose to column
    if depth_data.shape[0] == 1:
        depth_data = depth_data.T

    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(depth_data.shape, dtype=bool)
    else:
        mask = np.atleast_2d(mask)
        if mask.shape[0] == 1:
            mask = mask.T

    # Check if there are any masked (invalid) regions
    has_masked_regions = np.any(~mask)

    # Calculate Gaussian sigma from cutoff
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_lo, xdim)

    # Set up 2D sigma: filter only along first dimension (rows)
    if depth_data.shape[1] > 1:
        # 2D data: filter along rows only (perpendicular to striations)
        sigma_2d = (sigma, 0)
    else:
        # 1D profile
        sigma_2d = (sigma, 0)

    radius = (int(ceil(sigma)) + 1, 1)

    # Apply Gaussian lowpass filter
    if has_masked_regions:
        # Use weighted filtering for masked data
        # Set masked regions to NaN for proper handling
        depth_with_nan = depth_data.astype(float).copy()
        depth_with_nan[~mask] = np.nan
        filtered = _apply_nan_weighted_gaussian(depth_with_nan, sigma_2d, radius)
        mask_filtered = mask.copy()
    else:
        # No mask, use standard filtering
        filtered = _apply_nan_weighted_gaussian(
            depth_data.astype(float), sigma_2d, radius
        )
        mask_filtered = mask.copy()

    # For noise removal, return the smoothed data directly (not residuals)
    smoothed = filtered

    # Crop borders if requested
    sigma_int = int(ceil(sigma))

    if cut_borders_after_smoothing:
        if has_masked_regions:
            # Set invalid regions to NaN before cropping
            smoothed[~mask_filtered] = np.nan
            # Remove zero/NaN borders
            cropped_data, cropped_mask, range_indices = _remove_zero_border(
                smoothed, mask_filtered
            )
        else:
            # Simple border crop: remove sigma pixels from top and bottom
            if sigma_int > 0 and depth_data.shape[0] > 2 * sigma_int:
                cropped_data = smoothed[sigma_int:-sigma_int, :]
                cropped_mask = mask_filtered[sigma_int:-sigma_int, :]
                range_indices = np.arange(sigma_int, depth_data.shape[0] - sigma_int)
            else:
                # Data too small to crop
                cropped_data = smoothed
                cropped_mask = mask_filtered
                range_indices = np.arange(depth_data.shape[0])
    else:
        # No cropping
        if has_masked_regions:
            mask_filtered = mask_filtered  # Keep filtered mask
        cropped_data = smoothed
        cropped_mask = mask_filtered
        range_indices = np.arange(depth_data.shape[0])

    return NoiseRemovalResult(
        depth_data=cropped_data,
        range_indices=range_indices,
        mask=cropped_mask,
    )
