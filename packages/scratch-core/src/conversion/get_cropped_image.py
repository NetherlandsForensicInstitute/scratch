import numpy as np
from numpy.typing import NDArray

from conversion.gaussian_filter import apply_gaussian_filter
from conversion.leveling import SurfaceTerms, level_map
from conversion.mask import mask_and_crop_scan_image
from conversion.resample import resample_scan_image_and_mask
from utils.array_definitions import MaskArray
from image_generation.data_formats import ScanImage


def get_cropped_image(
    scan_image: ScanImage,
    mask: MaskArray,
    terms: SurfaceTerms,
    cutoff_length: float,
    regression_order: int = 0,
    resampling_factors: tuple[float, float] | None = None,
    crop: bool = False,
) -> NDArray:
    """
    Generate a preview image for the cropping editor by applying resampling, leveling, and filtering to depth data.

    :param scan_image: ScanImage to be processed.
    :param mask: Mask indicating fore/background to be applied to the data in `scan_image`.
    :param terms: The surface terms to be used in the fitting. Note: terms can be combined using bit-operators.
    :param cutoff_length: Cutoff wavelength in physical units.
    :param regression_order: Filter regression order used when filtering the data.
    :param resampling_factors: The resampling factors for the x- and y-axis.
    :param crop: Whether to crop the result (i.e. remove outer NaNs).
    :returns: A numpy array with the cropped image data.
    """
    # Check whether the mask only consists of background
    if not np.any(mask):
        return np.full_like(scan_image.data, np.nan)

    # Resample image and mask to speed up the processing
    resampled_scan_image, resampled_mask = resample_scan_image_and_mask(
        scan_image, mask, resampling_factors=resampling_factors
    )

    # Apply mask to the `ScanImage` instance
    if resampled_mask is not None:
        resampled_scan_image = mask_and_crop_scan_image(
            scan_image=resampled_scan_image, mask=resampled_mask, crop=crop
        )

    # Level the image data
    level_result = level_map(scan_image=resampled_scan_image, terms=terms)

    # Filter the leveled results using a Gaussian filter
    data_filtered = apply_gaussian_filter(
        data=level_result.leveled_map,
        regression_order=regression_order,
        cutoff_length=cutoff_length,
        pixel_size=(resampled_scan_image.scale_x, resampled_scan_image.scale_y),
        is_high_pass=True,
    )

    return data_filtered
