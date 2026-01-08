"""
Remove large-scale shape/form from surface data using Gaussian highpass filter.

This module provides functionality to remove the large-scale form (curvature,
tilt, waviness) from surface scan data, leaving only the fine-scale features
such as striation marks. This is achieved using a Gaussian highpass filter.

The highpass filter works by:
1. Applying a Gaussian lowpass filter to extract the shape
2. Subtracting the shape from the original data (residuals = original - smoothed)
"""

from functools import partial
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from conversion.preprocessing.cheby_cutoff_to_gauss_sigma import (
    cheby_cutoff_to_gauss_sigma,
)


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


def remove_shape_gaussian(
    depth_data: NDArray[np.floating],
    xdim: float,
    cutoff_hi: float = 2000.0,
    cut_borders_after_smoothing: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.intp], NDArray[np.bool_]]:
    """
    Remove toolmark/surface shape using Gaussian highpass filter.

    This function removes large-scale form (curvature, tilt, waviness) from
    surface scan data using a Gaussian highpass filter. The result contains
    only high-frequency features like striation marks.

    The filtering is applied only along the first dimension (rows), which
    corresponds to the direction perpendicular to striation marks. This
    preserves striation features while removing shape variations.

    Algorithm:
        1. Convert cutoff wavelength to Gaussian sigma using ISO standard
        2. Apply 1D Gaussian lowpass filter along rows to extract shape
        3. Compute residuals: output = input - smoothed (highpass)
        4. Optionally crop border artifacts (sigma pixels from each edge)

    Parameters
    ----------
    depth_data : NDArray[np.floating]
        2D depth/height data array, or 1D profile. For 2D data,
        rows should be perpendicular to striation direction.
    xdim : float
        Pixel spacing in meters (m). Distance between adjacent
        measurements in the scan.
    cutoff_hi : float, optional
        High-frequency cutoff wavelength in micrometers (um).
        Larger values remove more of the surface form. Default 2000 um is
        typical for shape removal while preserving striation features.
    cut_borders_after_smoothing : bool, optional
        If True, crop ceil(sigma) pixels from top and bottom edges to
        remove filter artifacts. Default True.
    mask : NDArray[np.bool_] | None, optional
        Boolean mask array (True = valid data). If provided,
        masked regions are excluded from filtering. Must match depth_data shape.

    Returns
    -------
    depth_data : NDArray[np.floating]
        Filtered data with shape removed (residuals). Contains the
        high-frequency content (striation marks).
    range_indices : NDArray[np.intp]
        Array of row indices that are valid after cropping. If
        cut_borders_after_smoothing=True, indicates which rows from
        the original data are included in the output.
    mask : NDArray[np.bool_]
        Boolean mask indicating valid data points in the output.
        True = valid data, False = invalid/masked region.

    Example
    -------
    >>> import numpy as np
    >>> x = np.linspace(0, 100, 500)
    >>> shape = 0.5 * x**2  # Parabolic shape
    >>> noise = np.random.randn(500) * 0.1  # Fine detail
    >>> surface = np.tile(shape + noise, (100, 1)).T
    >>> depth_data, range_indices, mask = remove_shape_gaussian(surface, xdim=1e-6)

    Notes
    -----
    - For 2D data, filtering is 1D along rows only (sigma_col = 0)
    - NaN values are handled using weighted filtering
    - The function matches MATLAB RemoveShapeGaussian.m behavior
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
    sigma = cheby_cutoff_to_gauss_sigma(cutoff_hi, xdim)

    # Set up 2D sigma: filter only along first dimension (rows)
    # For 2D data: sigma_2d = [0, sigma] means no filtering in row direction,
    # sigma filtering in column direction. But MATLAB uses [0, sigma] differently.
    # In MATLAB RemoveShapeGaussian.m: sigma_2d = [0 sigma] for 2D data
    # scipy convention: sigma[0] is for axis 0 (rows), sigma[1] is for axis 1 (cols)
    # MATLAB filters along columns (axis 1 in row-major), so we use sigma for axis 0
    if depth_data.shape[1] > 1:
        # 2D data: filter along rows only (perpendicular to striations)
        sigma_2d = (sigma, 0)
    else:
        # 1D profile
        sigma_2d = (sigma, 0)

    radius = (int(ceil(sigma)) + 1, 1)

    # Apply Gaussian lowpass filter (extracts shape)
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

    # Compute residuals (highpass = original - lowpass)
    residuals = depth_data - filtered

    # Crop borders if requested
    sigma_int = int(ceil(sigma))

    if cut_borders_after_smoothing:
        if has_masked_regions:
            # Set invalid regions to NaN before cropping
            residuals[~mask_filtered] = np.nan
            # Remove zero/NaN borders
            cropped_data, cropped_mask, range_indices = _remove_zero_border(
                residuals, mask_filtered
            )
        else:
            # Simple border crop: remove sigma pixels from top and bottom
            if sigma_int > 0 and depth_data.shape[0] > 2 * sigma_int:
                cropped_data = residuals[sigma_int:-sigma_int, :]
                cropped_mask = mask_filtered[sigma_int:-sigma_int, :]
                range_indices = np.arange(sigma_int, depth_data.shape[0] - sigma_int)
            else:
                # Data too small to crop
                cropped_data = residuals
                cropped_mask = mask_filtered
                range_indices = np.arange(depth_data.shape[0])
    else:
        # No cropping
        if has_masked_regions:
            mask_filtered = mask_filtered  # Keep filtered mask
        cropped_data = residuals
        cropped_mask = mask_filtered
        range_indices = np.arange(depth_data.shape[0])

    return cropped_data, range_indices, cropped_mask
