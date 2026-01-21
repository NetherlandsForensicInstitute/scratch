import numpy as np
from scipy.ndimage import generic_filter

from container_models.base import MaskArray, ScanMap2DArray
from container_models.scan_image import ScanImage
from conversion.mask import mask_2d_array
from conversion.resample import resample_scan_image_and_mask
from conversion.utils import unwrap_result, update_scan_image_data
from parsers import subsample_scan_image

# Downsample goal in micrometers to make filter computations faster
DOWNSAMPLE_GOAL = 70e-6
MEDIAN_FILTER_SIZE = 5
SMALL_STRIP_THRESHOLD = 20
TIMES_MEDIAN_CORRECTION_FACTOR = 6


def mask_and_remove_needles(
    scan_image: ScanImage, mask: MaskArray, times_median: float = 15.0
) -> ScanImage:
    """
    Mask the scan image and remove needle artifacts (i.e. steep slopes) using median filtering.

    Implements the following flow:
    - Apply the mask to the scan image to exclude any outliers/needles that are irrelevant.
    - Apply median filtering to smooth the data. If the image is large, it is downsampled before filtering and
        upsampled afterward. If the image is a small strip of data (width or height <= SMALL_STRIP_THRESHOLD), the
        filter size is reduced.
    - Compute residuals as the difference between the masked original and median filtered data.
    - Mark points as needles where residuals exceed a threshold (absolute data median * times_median *
        TIMES_MEDIAN_CORRECTION_FACTOR)
    - Set marked needle points to NaN in the masked original scan image

    :param scan_image: Scan image to mask and clean.
    :param mask: Binary mask array.
    :param times_median: Parameter to help determine the needle threshold.
    :return: The masked and cleaned scan image.
    """
    scan_image_masked = update_scan_image_data(
        scan_image, mask_2d_array(scan_image.data, mask)
    )

    residual_image = get_residual_image(scan_image_masked)

    return determine_and_remove_needles(scan_image_masked, residual_image, times_median)


def get_residual_image(scan_image: ScanImage) -> ScanMap2DArray:
    """
    Apply median filtering to smooth the image and compute residuals as the difference between the input scan image and
    the median filtered image.

    If the image is large, it is downsampled before filtering and upsampled afterward. If the image is a small strip of
    data (width or height <= SMALL_STRIP_THRESHOLD), the filter size is reduced to avoid too extensive smoothing.

    :param scan_image: Scan image to calculate residual image for. Assumes any masks have already been applied.
    :return: Array of differences between the input scan_image.data and median filter smoothed version of that image.
    """
    # Check if the image is a small strip of data
    is_small_strip = (
        scan_image.width <= SMALL_STRIP_THRESHOLD
        or scan_image.height <= SMALL_STRIP_THRESHOLD
    )

    if not is_small_strip:
        # Calculate subsampling factor for computational efficiency using the given DOWNSAMPLE_GOAL and the desired
        # MEDIAN_FILTER_SIZE
        subsample_factor = int(
            np.ceil(DOWNSAMPLE_GOAL / MEDIAN_FILTER_SIZE / scan_image.scale_x)
        )

        # If a subsample_factor is defined, downsample the data before filtering and upsample back after filtering.
        # Otherwise, just apply the filter directly.
        if subsample_factor > 1:
            scan_image_subsampled = unwrap_result(
                subsample_scan_image(
                    scan_image=scan_image,
                    step_size_x=subsample_factor,
                    step_size_y=subsample_factor,
                )
            )

            scan_image_subsampled_filtered = apply_median_filter(
                scan_image=scan_image_subsampled, filter_size=MEDIAN_FILTER_SIZE
            )

            upsample_factors = (1 / subsample_factor, 1 / subsample_factor)
            scan_image_filtered, _ = resample_scan_image_and_mask(
                scan_image=scan_image_subsampled_filtered,
                factors=upsample_factors,
                only_downsample=False,
            )

        else:
            scan_image_filtered = apply_median_filter(
                scan_image=scan_image, filter_size=MEDIAN_FILTER_SIZE
            )

        # Use slicing since the shape may deviate slightly after down- and upsampling
        residual_image = (
            scan_image.data
            - scan_image_filtered.data[
                : scan_image.data.shape[0], : scan_image.data.shape[1]
            ]
        )
    else:
        filter_size_adjusted = int(np.round(np.sqrt(MEDIAN_FILTER_SIZE)))
        scan_image_filtered = apply_median_filter(
            scan_image=scan_image, filter_size=filter_size_adjusted
        )

        # Handle transposition for single-row data
        if scan_image_filtered.width == 1:
            residual_image = scan_image.data - scan_image_filtered.data.T
        else:
            residual_image = scan_image.data - scan_image_filtered.data

    return residual_image


def determine_and_remove_needles(
    scan_image: ScanImage, residual_image: ScanMap2DArray, times_median: float
) -> ScanImage:
    """
    Mark points as needles where residuals exceed a threshold (determined as the absolute median of the residuals *
    times_median * TIMES_MEDIAN_CORRECTION_FACTOR) and set marked needle points to NaN in scan image.

    :param scan_image: ScanImage to remove needles from. Assumes any masks are already applied.
    :param residual_image: Array of differences between a (masked) image and median filter smoothed image.
    :param times_median: Parameter to help determine the needle threshold.
    :return: ScanImage where any needles are removed.
    """
    times_median = times_median * TIMES_MEDIAN_CORRECTION_FACTOR

    # Find needles: points where |residual| > threshold
    median_residual = np.nanmedian(np.abs(residual_image))
    threshold = times_median * (1 if median_residual == 0 else median_residual)
    needles_indices = np.abs(residual_image) > threshold

    # Remove needles from the image by setting them to NaN
    data_without_needles = scan_image.data.copy()
    data_without_needles[needles_indices] = np.nan

    return update_scan_image_data(scan_image, data_without_needles)


def apply_median_filter(scan_image: ScanImage, filter_size: int) -> ScanImage:
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

    return update_scan_image_data(scan_image, filtered_image)
