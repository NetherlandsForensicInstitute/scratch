import numpy as np
from scipy.ndimage import generic_filter

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.mask import mask_2d_array
from conversion.resample import resample_scan_image_and_mask
from conversion.utils import unwrap_result
from parsers import subsample_scan_image

FILTER_SIZE_MODERATED = 5


def remove_needles(
    scan_image: ScanImage, mask: MaskArray, times_median: float = 15.0
) -> ScanImage:
    """
    Remove needle artifacts (outliers) from depth measurement data using median filtering.

    This function identifies and removes spike-like outliers in depth data by:
    1. Applying median filtering to smooth the data
    2. Computing residuals (difference between original and smoothed data)
    3. Flagging points where residuals exceed a threshold based on median absolute deviation
    4. Setting flagged outlier points to NaN

    The function adapts its filtering strategy based on data size:
    - For large datasets (>20 columns or rows): uses 2D median filtering with optional subsampling
    - For small datasets (≤20 columns or rows): uses 1D median filtering with reduced filter size

    :param scan_image: Scan image to clean.
    :param mask: Binary mask array.
    :param times_median: Parameter to help determine the outlier threshold.
    :return: The cleaned scan image.
    """
    times_median = times_median * 6

    # Check if this is a small strip of data
    is_small_strip = scan_image.width <= 20 or scan_image.height <= 20

    if not is_small_strip:
        # Calculate subsampling factor for computational efficiency
        # Goal: 7 μm sampling with 70 μm filter diameter
        subsample_factor = int(
            np.ceil(70e-6 / FILTER_SIZE_MODERATED / scan_image.scale_x)
        )

        # Apply mask and prepare data
        scan_image_masked = ScanImage(
            data=mask_2d_array(scan_image.data, mask),
            scale_x=scan_image.scale_x,
            scale_y=scan_image.scale_y,
        )

        if subsample_factor > 1:
            scan_image_subsampled = unwrap_result(
                subsample_scan_image(
                    scan_image_masked, subsample_factor, subsample_factor
                )
            )
            # Apply median filter (using nanmedian equivalent)
            scan_image_subsampled_filtered = apply_median_filter(
                scan_image_subsampled, FILTER_SIZE_MODERATED
            )
            # Upsample back to original resolution
            upsample_factors = (1 / subsample_factor, 1 / subsample_factor)
            scan_image_filtered, _ = resample_scan_image_and_mask(
                scan_image_subsampled_filtered,
                factors=upsample_factors,
                only_downsample=False,
            )

        else:
            # Apply median filter (using nanmedian equivalent)
            scan_image_filtered = apply_median_filter(
                scan_image_masked, FILTER_SIZE_MODERATED
            )

        residual_image = (
            scan_image_masked.data
            - scan_image_filtered.data[
                : scan_image.data.shape[0], : scan_image.data.shape[1]
            ]
        )
    else:
        # For small strips: use 1D filtering with adjusted kernel size
        # Convert 2D filter size to 1D equivalent: sqrt(10) ≈ 3
        filter_size_adjusted = int(np.round(np.sqrt(FILTER_SIZE_MODERATED)))

        scan_image_filtered = apply_median_filter(scan_image, filter_size_adjusted)

        # Handle transposition for single-row data
        if scan_image_filtered.width == 1:
            residual_image = scan_image.data - scan_image_filtered.data.T
        else:
            residual_image = scan_image.data - scan_image_filtered.data

    # Find outliers: points where |residual| > threshold
    threshold = times_median * np.nanmedian(np.abs(residual_image))
    indices_invalid = np.abs(residual_image) > threshold

    # Remove outliers by setting them to NaN
    scan_image_without_outliers = scan_image.data.copy()
    scan_image_without_outliers[indices_invalid] = np.nan

    return ScanImage(
        data=scan_image_without_outliers,
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )


def apply_median_filter(scan_image: ScanImage, filter_size: int):
    """
    Apply a median filter with NaN handling to scan_image.data.

    This function computes the median value within a sliding window for each pixel, ignoring NaN values in the
    computation. Pixels near image boundaries are handled by padding with NaN values, which are also ignored during
    median calculation.

    Median filters conventionally use odd-sized windows, thus the filter_size is corrected to be an odd number
    for the following reasons:
    - Odd dimensions ensure a clear center pixel with equal numbers of neighbors on all sides.
    - Even filter sizes would produce asymmetric filtering with bias toward upper-left pixels, leading to unexpected and
    inconsistent results.

    :param scan_image: The scan image to filter.
    :param filter_size: Size of the square median filter window. Will be adjusted to the next odd
        integer if an even value is provided (e.g., 4 becomes 5, 6 becomes 7).
        Must be positive.
    :return: Scan image with filtered data, with same shape and scales as input data
    """
    if filter_size % 2 == 0:
        filter_size += 1

    filtered_image = generic_filter(
        scan_image.data, np.nanmedian, size=filter_size, mode="constant", cval=np.nan
    ).astype(np.float64)

    return ScanImage(
        data=filtered_image, scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
    )
