import numpy as np

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.mask import mask_2d_array
from conversion.resample import resample_scan_image_and_mask
from parsers import subsample_scan_image


def remove_needles(
    scan_image: ScanImage, mask: MaskArray, times_median: int = 15
) -> ScanImage:
    """
    Remove needle artifacts (outliers) from depth measurement data using median filtering.

    This function identifies and removes spike-like outliers in depth data by:
    1. Applying median filtering to smooth the data
    2. Computing residuals (difference between original and smoothed data)
    3. Flagging points where residuals exceed a threshold based on median absolute deviation
    4. Setting flagged outlier points to NaN

    The function adapts its filtering strategy based on data size:
    - For large datasets (>20 columns): uses 2D median filtering with optional subsampling
    - For small datasets (≤20 columns): uses 1D median filtering with reduced filter size

    :param scan_image:
    :param mask:
    :param times_median:
    :return:
    """
    filter_size_moderated = 5
    times_median = times_median * 6

    # Check if this is a small strip of data
    is_small_strip = scan_image.width <= 20

    if not is_small_strip:
        # Calculate subsampling factor for computational efficiency
        # Goal: 7 μm sampling with 70 μm filter diameter
        subsample_factor = int(
            np.ceil(70e-6 / filter_size_moderated / scan_image.scale_x)
        )

        # Apply mask and prepare data
        scan_image_masked = ScanImage(
            data=mask_2d_array(scan_image.data, mask),
            scale_x=scan_image.scale_x,
            scale_y=scan_image.scale_y,
        )

        # Subsample if needed
        if subsample_factor > 1:
            scan_image_subsampled = subsample_scan_image(
                scan_image_masked, subsample_factor, subsample_factor
            )
            # Apply median filter (using nanmedian equivalent)
            scan_image_subsampled_filtered = apply_median_filter(
                scan_image_subsampled, filter_size_moderated
            )
            # Upsample back to original resolution
            upsample_factors = (1 / subsample_factor, 1 / subsample_factor)
            scan_image_filtered = resample_scan_image_and_mask(
                scan_image_subsampled_filtered,
                factors=upsample_factors,
                only_downsample=False,
            )

        else:
            # Apply median filter (using nanmedian equivalent)
            scan_image_filtered = apply_median_filter(
                scan_image_masked, filter_size_moderated
            )

        # Compute residuals
        residual_image = (
            scan_image_masked.data
            - scan_image_filtered.data[
                : scan_image.data.shape[0], : scan_image.data.shape[1]
            ]
        )
    else:
        # For small strips: use 1D filtering with adjusted kernel size
        # Convert 2D filter size to 1D equivalent: sqrt(10) ≈ 3
        filter_size_adjusted = int(np.round(np.sqrt(filter_size_moderated)))

        scan_image_filtered = apply_median_filter(scan_image, filter_size_adjusted)

        # Handle transposition for single-row data
        if scan_image_filtered.data.shape[0] == 1:
            residual_image = scan_image.data - scan_image_filtered.data.T
        else:
            residual_image = scan_image.data - scan_image_filtered.data

    # Find outliers: points where |residual| > threshold
    threshold = times_median * np.nanmedian(np.abs(residual_image))
    indices_invalid = np.abs(residual_image) > threshold

    # Remove outliers by setting them to NaN
    scan_image_filtered = scan_image.data.copy()
    scan_image_filtered[indices_invalid] = np.nan

    return scan_image_filtered


def apply_median_filter(scan_image: ScanImage, filter_size: int) -> ScanImage:
    """
    Apply a fast median filter that handles NaN values.

    This function computes a median filter by creating shifted versions of the input
    image and taking the median across all shifts. NaN values are ignored during
    the median calculation.

    Notes
    -----
    This implementation uses circular shifts to create a 3D array of all
    neighborhood pixels, then computes the median along the third dimension.

    :param scan_image: Scan image to filter
    :param filter_size: Size of the median filter kernel (will be made odd if even)
    :return: Median-filtered scan image with the same shape as input_image
    """
    # Make sure the filtersize is odd
    if filter_size % 2 == 0:
        filter_size = filter_size + 1

    half_filt_size = (filter_size - 1) // 2

    # Pad the matrix with border_mult on all sides
    pad_shape = (
        scan_image.data.shape[0] + filter_size - 1,
        scan_image.data.shape[1] + filter_size - 1,
    )
    input_image_border = np.ones(pad_shape) * np.nan
    input_image_border[
        half_filt_size:-half_filt_size, half_filt_size:-half_filt_size
    ] = scan_image.data

    # Create 3D array to hold all shifted versions
    input_image_array = np.ones((*pad_shape, filter_size**2), dtype=np.float32)

    # Fill the array with circularly shifted versions
    image_count = 0
    for kernel_rows in range(-half_filt_size, half_filt_size + 1):
        for kernel_columns in range(-half_filt_size, half_filt_size + 1):
            input_image_array[:, :, image_count] = np.roll(
                input_image_border, shift=(kernel_rows, kernel_columns), axis=(0, 1)
            )
            image_count += 1

    # Remove borders and compute median
    output_image_no_border = input_image_array[
        half_filt_size:-half_filt_size, half_filt_size:-half_filt_size, :
    ]
    output_image = np.nanmedian(output_image_no_border, axis=2)

    return ScanImage(
        data=output_image, scale_x=scan_image.scale_x, scale_y=scan_image.scale_y
    )
