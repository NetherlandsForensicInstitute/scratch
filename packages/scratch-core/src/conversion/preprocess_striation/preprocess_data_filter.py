from math import ceil
from scipy import ndimage
from functools import partial

import numpy as np
from numpy.typing import NDArray

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.mask import _determine_bounding_box


def _apply_nan_weighted_gaussian_1d(
    data: NDArray[np.floating],
    sigma: float,
    truncate: float = 4.0,
) -> NDArray[np.floating]:
    """
    Apply 1D Gaussian filter along rows with NaN-aware weighting.

    NaN values are excluded from the convolution by setting their weight to 0.
    The result is normalized by the sum of weights to compensate for missing data.

    :param data: Input 2D data array (may contain NaN values).
    :param sigma: Standard deviation of Gaussian kernel (in pixels).
    :param truncate: Truncate filter at this many standard deviations (default 4.0).
    :returns: Filtered data array. NaN positions will have interpolated values
        based on neighboring valid data.
    """

    gaussian_filter = partial(
        ndimage.gaussian_filter,
        sigma=(sigma, 0),  # Filter only along first axis (rows)
        mode="reflect",  # Match MATLAB's symmetric boundary conditions
        truncate=truncate,
    )

    nan_mask = np.isnan(data)
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


def cheby_cutoff_to_gauss_sigma(cutoff: float, pixel_size: float) -> float:
    """
    Convert cutoff wavelength to Gaussian sigma using ISO 16610 standard.

    :param cutoff: Cutoff wavelength in physical units (e.g., meters).
    :param pixel_size: Pixel spacing in the same units as cutoff.
    :return: Gaussian sigma in pixel units.
    """
    alpha_gaussian = np.sqrt(2 * np.log(2)) / (2 * np.pi)
    cutoff_pixels = cutoff / pixel_size
    return float(cutoff_pixels * alpha_gaussian)


def apply_gaussian_filter_1d(
    scan_image: ScanImage,
    cutoff: float,
    is_high_pass: bool = False,
    cut_borders_after_smoothing: bool = True,
    mask: MaskArray | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.intp], MaskArray]:
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

    :param scan_image: ScanImage containing depth data and pixel spacing.
    :param cutoff: Cutoff wavelength in meters (m).
    :param is_high_pass: If False, returns smoothed data (lowpass). If True, returns residuals (highpass).
    :param cut_borders_after_smoothing: If True, crop ceil(sigma) pixels from top and bottom edges.
    :param mask: Boolean mask array (True = valid data). Must match depth_data shape.

    :returns filtered_data: Filtered data.
    :returns range_indices: Array of row indices that are valid after cropping.
    :returns mask: Boolean mask indicating valid data points in the output.
    """
    # Initialize mask if not provided
    if mask is None:
        mask = np.ones(scan_image.data.shape, dtype=bool)

    # Calculate Gaussian sigma from cutoff
    sigma = cheby_cutoff_to_gauss_sigma(cutoff, scan_image.scale_x)

    # Apply Gaussian lowpass filter, Use weighted filtering for masked data
    depth_with_nan = scan_image.data.copy()
    depth_with_nan[~mask] = np.nan
    filtered = _apply_nan_weighted_gaussian_1d(depth_with_nan, sigma)

    # Compute output based on filter type
    if is_high_pass:
        # Highpass: return residuals (original - smoothed)
        output = scan_image.data - filtered
    else:
        # Lowpass: return smoothed data directly
        output = filtered

    # Crop borders if requested
    sigma_int = int(ceil(sigma))

    # Check if there are any masked (invalid) regions
    has_masked_regions = np.any(~mask)

    if cut_borders_after_smoothing and has_masked_regions:
        output_with_nan = output.copy()
        output_with_nan[~mask] = np.nan
        # Remove zero/NaN borders
        cropped_data, cropped_mask, range_indices = _remove_zero_border(
            output_with_nan, mask
        )
    elif cut_borders_after_smoothing and (
        sigma_int > 0 and scan_image.data.shape[0] > 2 * sigma_int
    ):
        cropped_data = output[sigma_int:-sigma_int, :]
        cropped_mask = mask[sigma_int:-sigma_int, :]
        range_indices = np.arange(sigma_int, scan_image.data.shape[0] - sigma_int)
    else:
        # Data too small to crop so no cropping
        cropped_data = output
        cropped_mask = mask
        range_indices = np.arange(scan_image.data.shape[0])

    return cropped_data, range_indices, cropped_mask
