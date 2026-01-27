from container_models.base import ScanMap2DArray
import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from container_models.scan_image import ScanImage
from conversion.filter import (
    _apply_order0_filter,
    _apply_polynomial_filter,
    _create_normalized_separable_kernels,
)
from conversion.leveling import SurfaceTerms, level_map
from conversion.resample import _clip_factors

# TODO: Based on what this code is doing. We can ignore this function completely
# [] First breakdown the code smells below, make sure they give the same response
# [x] extract resample_scan_image, resample_array (mask_and_crop_scan_image)
# [x] extract mask_scan_image, crop_scan_image (mask_and_crop_scan_image)
# [] extract level_map
# [] extroct apply_gaussian_regression_filter
# [] Remove get_cropped_image


# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
# alpha = Sqrt((-1 - LambertW(-1, -1 / (2 * exp(1)))) / Pi)
ALPHA_REGRESSION = 0.7309134280946760


def apply_gaussian_regression_filter(
    data: NDArray[np.floating],
    cutoff_pixels: NDArray[np.floating],
    regression_order: int,
    nan_out: bool,
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

    return smoothed


def resample_scan_image(image: ScanImage, factors: tuple[float, float]) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The resampled ScanImage.
    """
    mask = (
        None
        if image.mask is None
        else resample_image_array(image.mask, factors=factors)
    )
    return ScanImage(
        data=resample_image_array(image.data, factors=factors),
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
        mask=mask,
    )


def resample_image_array(
    array: NDArray,
    factors: tuple[float, float],
) -> NDArray:
    """
    Resample an array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    resampled = resize(
        image=array,
        output_shape=(1 / factor_y * array.shape[0], 1 / factor_x * array.shape[1]),
        mode="edge",
        anti_aliasing=array.dtype != np.bool_ and all(factor > 1 for factor in factors),
    )
    return np.asarray(resampled, dtype=array.dtype)


def mask_image(image: ScanImage) -> ScanImage:
    """Apply the mask to the data."""
    image = image.model_copy()
    image.data[~image.mask] = np.nan  # type: ignore[index]
    return image


def _determine_bounding_box(mask: ScanMap2DArray) -> tuple[slice, slice]:
    """
    Determines the bounding box of non-zero values in a mask.

    :param mask: Binary mask array
    :return: Tuple of (y_slice, x_slice) for the bounding box
    """
    non_zero_coords = np.nonzero(mask)
    if not non_zero_coords[0].size:
        raise ValueError("Mask is empty")

    y_min, x_min = np.min(non_zero_coords, axis=1)
    y_max, x_max = np.max(non_zero_coords, axis=1)
    return slice(x_min, x_max + 1), slice(y_min, y_max + 1)


def crop_to_mask(image: ScanImage) -> ScanImage:
    """Crop to the bounding box of the mask."""
    if image.mask is None:
        raise ValueError("Mask is required for cropping operation.")
    y_slice, x_slice = _determine_bounding_box(image.mask)
    return image.model_copy(
        update={
            "data": image.data[y_slice, x_slice],
            "mask": None if image.mask is None else image.mask[y_slice, x_slice],
        }
    )


def get_cropped_image(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    cutoff_length: float,
    regression_order: int,
    resampling_factors: tuple[float, float],
    crop: bool = False,
) -> NDArray:
    """
    Generate a preview image for the cropping editor by applying resampling, leveling, and filtering to depth data.

    :param scan_image: ScanImage to be processed.
    :param mask: Mask indicating fore/background to be applied to the data in `scan_image`.
    :param terms: The surface terms to be used in the fitting. Note: terms can be combined using bit-operators.
    :param cutoff_length: Cutoff wavelength in physical units.
    :param regression_order: Filter regression order used when filtering the data.
    :param resampling_factors: The resampling factors for the X- and Y-axis scales.
    :param crop: Whether to crop the result (i.e. remove outer NaNs).
    :returns: A numpy array with the cropped image data.
    """
    resampling_factors = _clip_factors(resampling_factors, True)
    scan_image = resample_scan_image(image=scan_image, factors=resampling_factors)
    if scan_image.mask is None:
        raise ValueError("Mask is required for cropping operation.")

    scan_image = mask_image(scan_image)
    if crop:
        scan_image = crop_to_mask(scan_image)

    level_result = level_map(scan_image=scan_image, terms=terms)

    # Filter the leveled results using a Gaussian regression filter
    data_filtered = apply_gaussian_regression_filter(
        data=level_result.leveled_map,
        regression_order=regression_order,
        cutoff_pixels=cutoff_length
        / np.array([scan_image.scale_x, scan_image.scale_y]),
        nan_out=True,
        is_high_pass=True,
    )

    return data_filtered
