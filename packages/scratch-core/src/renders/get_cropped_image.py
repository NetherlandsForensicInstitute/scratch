import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from container_models.scan_image import ScanImage
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling import SurfaceTerms, level_map
from conversion.mask import mask_and_crop_scan_image
from conversion.resample import _clip_factors

# TODO: Based on what this code is doing. We can ignore this function completely
# [] First breakdown the code smells below, make sure they give the same response
# [x] extract resample_scan_image, resample_array (mask_and_crop_scan_image)
# [] extract mask_scan_image, crop_scan_image (mask_and_crop_scan_image)
# [] extract level_map
# [] extroct apply_gaussian_regression_filter
# [] Remove get_cropped_image


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
    resampled_scan_image = resample_scan_image(
        image=scan_image, factors=resampling_factors
    )

    # Apply mask to the `ScanImage` instance
    # Function is a code smell, call each task seprately
    if resampled_scan_image.mask is not None:
        resampled_scan_image = mask_and_crop_scan_image(
            scan_image=resampled_scan_image, mask=resampled_scan_image.mask, crop=crop
        )

    # Level the image data
    level_result = level_map(scan_image=resampled_scan_image, terms=terms)

    # Filter the leveled results using a Gaussian regression filter
    data_filtered = apply_gaussian_regression_filter(
        data=level_result.leveled_map,
        regression_order=regression_order,
        cutoff_length=cutoff_length,
        pixel_size=(resampled_scan_image.scale_x, resampled_scan_image.scale_y),
        is_high_pass=True,
    )

    return data_filtered
