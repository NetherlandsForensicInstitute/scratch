import numpy as np
from numpy.typing import NDArray

from conversion.gaussian_filter import apply_gaussian_filter
from conversion.leveling import SurfaceTerms, level_map
from conversion.mask import mask_and_crop_scan_image
from utils.array_definitions import MaskArray
from image_generation.data_formats import ScanImage


# place-holder function
def resample_image_and_mask(
    scan_image: ScanImage, mask: MaskArray, resample_factors: tuple[float, float]
) -> tuple[ScanImage, MaskArray]:
    return scan_image, mask


def get_cropped_image(
    scan_image: ScanImage,
    mask: MaskArray,
    terms: SurfaceTerms,
    regression_order: int,
    cutoff_lengths: tuple[float, float],
    resample_factors: tuple[float, float] | None = None,
) -> NDArray:
    """
    Generate a preview image for the cropping editor by applying resampling, leveling, and filtering to depth data.

    :param scan_image: ScanImage to be processed.
    :param mask: Mask indicating fore/background to be applied to the data in `scan_image`.
    :param terms: The surface terms to be used in the fitting. Note: terms can be combined using bit-operators.
    :param regression_order: Filter regression order used when filtering the data.
    :param cutoff_lengths: Cutoff wavelengths in physical units.
    :param resample_factors: TODO
    :returns: A numpy array with the cropped image data.
    """
    # Check whether the mask only consists of background
    if not np.any(mask):
        return np.full_like(scan_image.data, np.nan)

    # Resample image and mask to speed up the processing
    resampled_image, resampled_mask = resample_image_and_mask(
        scan_image, mask, resample_factors=resample_factors
    )

    # Apply mask to the `ScanImage` instance
    masked_image = mask_and_crop_scan_image(
        scan_image=resampled_image, mask=resampled_mask, crop=False
    )

    # Level the image data
    level_result = level_map(scan_image=masked_image, terms=terms)

    # Filter the leveled results using a Gaussian filter
    data_filtered = apply_gaussian_filter(
        data=level_result.leveled_map,
        regression_order=regression_order,
        cutoff_lengths=cutoff_lengths,
        is_high_pass=True,
    )

    return data_filtered
