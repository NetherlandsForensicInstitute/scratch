from typing import Optional

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from conversion.data_formats import Mark
from container_models.scan_image import ScanImage
from container_models.base import BinaryMask, FloatArray2D


def resample_scan_image_and_mask(
    scan_image: ScanImage,
    mask: Optional[BinaryMask] = None,
    factors: Optional[tuple[float, float]] = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
) -> tuple[ScanImage, Optional[BinaryMask]]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser than the target scale,
    no resampling is performed. If `factors` are provided, it overrides the target scale.

    :param scan_image: Input ScanImage to resample.
    :param mask: Corresponding mask array.
    :param factors: The multipliers for the scale of the X- and Y-axis. The formula used is `new_scale = factor * old_scale`.
    :param target_scale: Target scale (in meters) when `factors` are not provided.
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data (default). If False, allow upsampling.
    :returns: Resampled ScanImage and MaskArray
    """
    if not factors:
        factors = get_scaling_factors(
            scales=(scan_image.scale_x, scan_image.scale_y), target_scale=target_scale
        )
    if only_downsample:
        factors = _clip_factors(factors, preserve_aspect_ratio)
    if np.allclose(factors, 1.0):
        return scan_image, mask
    image = _resample_scan_image(scan_image, factors=factors)
    if mask is not None:
        mask = resample_image_array(mask, factors=factors)
    return image, mask


def resample_mark(mark: Mark, only_downsample: bool = False) -> Mark:
    """Resample a Mark so that the scale matches the scale specific for the mark type.

    :param mark: The Mark to resample.
    :param only_downsample: If True, only resample if it would reduce the resolution.
    :returns: The resampled Mark.
    """
    resampled_scan_image, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=mark.mark_type.scale,
        only_downsample=only_downsample,
    )
    return mark.model_copy(update={"scan_image": resampled_scan_image})


def _resample_scan_image(image: ScanImage, factors: tuple[float, float]) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The resampled ScanImage.
    """
    image_array_resampled = resample_image_array(image.data, factors=factors)
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
    )


def resample_image_array(
    array: FloatArray2D | BinaryMask,
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


def get_scaling_factors(
    scales: tuple[float, float],
    target_scale: float,
) -> tuple[float, float]:
    """
    Calculate the multipliers for a target scale.

    :param scales: Current scales (= pixel size in meters per image dimension).
    :param target_scale: Target scale (= pixel size in meters).

    :returns: The computed multipliers.
    """
    return target_scale / scales[0], target_scale / scales[1]


def _clip_factors(
    factors: tuple[float, float],
    preserve_aspect_ratio: bool,
) -> tuple[float, float]:
    """Clip the scaling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        # Set the multipliers to equal values to preserve the aspect ratio
        max_factor = max(factors)
        factors = max_factor, max_factor

    return max(factors[0], 1.0), max(factors[1], 1.0)
