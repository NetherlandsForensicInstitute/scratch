from typing import Optional

import numpy as np
from numpydantic import NDArray
from scipy import ndimage

from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def resample_image_and_mask(
    image: ScanImage,
    mask: Optional[MaskArray] = None,
    resample_factors: Optional[tuple[float, float]] = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
) -> tuple[ScanImage, Optional[MaskArray]]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser
    than the target scale, no resampling is performed. If `resample_factors` are
    provided, it overrides the target scale.

    :param image: Input ScanImage to resample
    :param mask: Corresponding mask array
    :param resample_factors: Resampling factors
    :param target_scale: Target scale (m) when resample_factors are not provided
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data

    :returns: Resampled ScanImage and MaskArray
    """
    if not resample_factors:
        resample_factors = get_resampling_factors(
            image.scale_x,
            image.scale_y,
            target_scale,
        )
    if only_downsample:
        resample_factors = clip_resample_factors(
            resample_factors, only_downsample, preserve_aspect_ratio
        )
    if resample_factors == (1, 1):
        return image, mask

    image = resample_scan_image(image, resample_factors)
    if mask is not None:
        mask = resample_mask(mask, resample_factors)
    return image, mask


def resample_mask(mask: MaskArray, resample_factors: tuple[float, float]) -> MaskArray:
    """Resample the provided mask array using the specified resampling factors."""
    return _resample_array(mask, resample_factors, order=0, mode="nearest")


def resample_scan_image(
    image: ScanImage, resample_factors: tuple[float, float]
) -> ScanImage:
    """Resample the ScanImage object using the specified resampling factors."""
    image_array_resampled = _resample_array(
        image.data, resample_factors, order=1, mode="nearest"
    )
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x * resample_factors[0],
        scale_y=image.scale_y * resample_factors[1],
    )


def _resample_array(
    array: NDArray,
    resample_factors: tuple[float, float],
    order: int,
    mode: str,
) -> NDArray:
    """
    Resample an array using the specified resampling factors, order, and mode.

    :param array: The array to resample.
    :param resample_factors: The resampling factors for the x- and y-axis.
    :param order: The order of the spline interpolation to use.
    :param mode: The mode to use for handling boundaries.

    :returns: The resampled array.
    """
    resample_factor_x, resample_factor_y = resample_factors
    resampled = ndimage.zoom(
        array,
        (1 / resample_factor_y, 1 / resample_factor_x),
        order=order,
        mode=mode,
    )
    return np.asarray(resampled).astype(array.dtype)


def get_resampling_factors(
    scale_x: float,
    scale_y: float,
    target_scale: float,
) -> tuple[float, float]:
    """
    Calculate resampling factors for x and y dimensions. If `resample_factor` is provided, factors are set to 1/resample_factor,
    otherwise, factors are calculated from current scale and target_scale.

    The resampling factor determines how the image dimensions will change:
    - factor > 1: upsampling (more pixels, finer resolution)
    - factor < 1: downsampling (fewer pixels, coarser resolution)
    - factor = 1: no change

    :param scale_x: Scale for the x-axis
    :param scale_y: Scale for the y-axis
    :param target_scale: Target resolution (m). Used when `resample_factor` is None.

    :returns: Resampling factors.
    """
    resample_factor_x = target_scale / scale_x
    resample_factor_y = target_scale / scale_y
    return resample_factor_x, resample_factor_y


def clip_resample_factors(
    resample_factors: tuple[float, float],
    only_downsample: bool,
    preserve_aspect_ratio: bool,
) -> tuple[float, float]:
    if preserve_aspect_ratio:
        # Scale both factors equally to preserve the aspect ratio
        max_factor = max(resample_factors)
        resample_factors = (max_factor, max_factor)

    if only_downsample:
        # If only downsampling is allowed, clip the factors at max 1
        resample_factors = (
            max(resample_factors[0], 1.0),
            max(resample_factors[1], 1.0),
        )

    return resample_factors
