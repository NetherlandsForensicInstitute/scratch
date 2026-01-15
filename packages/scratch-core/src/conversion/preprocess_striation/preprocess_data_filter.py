from math import ceil

import numpy as np
from numpy.typing import NDArray

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.filter import apply_nan_weighted_gaussian_1d, cutoff_to_gaussian_sigma
from conversion.mask import _determine_bounding_box


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


def apply_gaussian_filter_1d(
    scan_image: ScanImage,
    cutoff: float,
    is_high_pass: bool = False,
    cut_borders_after_smoothing: bool = True,
    mask: MaskArray | None = None,
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
    output = apply_nan_weighted_gaussian_1d(
        data=depth_with_nan,
        cutoff_length=cutoff,
        pixel_size=scan_image.scale_x,
        axis=0,  # Filter along y-direction only
        is_high_pass=is_high_pass,
    )

    # Calculate sigma for border cropping
    sigma = cutoff_to_gaussian_sigma(cutoff, scan_image.scale_x)
    sigma_int = int(ceil(sigma))

    # Check if there are any masked (invalid) regions
    has_masked_regions = np.any(~mask)

    cropped_data = output
    cropped_mask = mask

    if cut_borders_after_smoothing:
        if has_masked_regions:
            output_with_nan = output.copy()
            output_with_nan[~mask] = np.nan
            cropped_data, cropped_mask, _ = _remove_zero_border(output_with_nan, mask)
        elif sigma_int > 0 and scan_image.height > 2 * sigma_int:
            cropped_data = output[sigma_int:-sigma_int, :]
            cropped_mask = mask[sigma_int:-sigma_int, :]

    return cropped_data, cropped_mask
