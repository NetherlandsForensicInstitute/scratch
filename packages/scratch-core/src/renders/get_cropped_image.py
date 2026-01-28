import numpy as np
from numpy.typing import NDArray
from parsers.samplers import resample_scan_image
from renders.filters import apply_gaussian_regression_filter
from renders.levelers import level_map
from utils.constants import RegressionOrder

from container_models.scan_image import ScanImage
from conversion.leveling import SurfaceTerms
from conversion.leveling.solver.utils import (
    compute_image_center,
)
from conversion.resample import _clip_factors

# TODO: Based on what this code is doing. We can ignore this function completely
# [] First breakdown the code smells below, make sure they give the same response
# [x] extract resample_scan_image, resample_array (mask_and_crop_scan_image)
# [x] extract mask_scan_image, crop_scan_image (mask_and_crop_scan_image)
# [x] extract level_map
# [x] extroct apply_gaussian_regression_filter
# [x]  move files to appropiate location
# []  add logging
# [] Remove get_cropped_image


def get_cropped_image(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    cutoff_length: float,
    regression_order: RegressionOrder,
    resampling_factors: tuple[float, float],
    crop: bool = False,
) -> NDArray:
    """
    Generate a preview image for the cropping editor by applying resampling, leveling, and filtering to depth data.

    :param scan_image: ScanImage to be processed. Must contain a mask.
    :param terms: The surface terms to be used in the polynomial fitting. Can be combined using bit-operators.
    :param cutoff_length: Cutoff wavelength in physical units for the Gaussian regression filter.
    :param regression_order: RegressionOrder enum specifying the filter polynomial order.
    :param resampling_factors: Tuple of (factor_x, factor_y) for resampling the X- and Y-axis scales.
    :param crop: If True, crop the result to the bounding box of the mask (removes outer NaNs).
    :returns: High-pass filtered image data (leveled - smoothed) as a numpy array.
    """
    resampling_factors = _clip_factors(resampling_factors, True)
    scan_image = resample_scan_image(image=scan_image, factors=resampling_factors)
    scan_image.apply_mask_image()

    if crop:
        scan_image.crop_to_mask()

    center_x, center_y = compute_image_center(scan_image)
    level_result = level_map(
        scan_image=scan_image, terms=terms, reference_point=(center_x, center_y)
    )

    # Filter the leveled results using a Gaussian regression filter
    filtered_data = apply_gaussian_regression_filter(
        data=level_result,
        regression_order=regression_order,
        cutoff_pixels=cutoff_length
        / np.array([scan_image.scale_x, scan_image.scale_y]),
    )
    filtered_data[np.isnan(level_result)] = np.nan
    return level_result - filtered_data
