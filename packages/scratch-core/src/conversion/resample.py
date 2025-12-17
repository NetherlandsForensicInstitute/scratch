from typing import Optional

import numpy as np
from numpydantic import NDArray
from scipy import ndimage

from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def resample_image_and_mask(
    image: ScanImage,
    mask: Optional[MaskArray] = None,
    resample_factor: Optional[float] = None,
    target_resolution: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = False,
) -> tuple[ScanImage, Optional[MaskArray]]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser
    than the target resolution, no resampling is performed. If `resample_factor` is
    provided, it overrides the target resolution.

    :param image: Input ScanImage to resample
    :param mask: Corresponding mask array
    :param resample_factor: Resampling factor (1/resample_factor is applied). If None, resamples to target_resolution.
    :param target_resolution: Target resolution (m) when resample_factor is not provided
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data

    :returns: Resampled ScanImage and MaskArray
    """
    resample_factor_x, resample_factor_y = get_resampling_factors(
        image.scale_x,
        image.scale_y,
        only_downsample,
        resample_factor,
        target_resolution,
        preserve_aspect_ratio,
    )
    if resample_factor_x == 1 and resample_factor_y == 1:
        return image, mask

    image = resample_scan_image(image, resample_factor_x, resample_factor_y)
    if mask:
        mask = resample_mask(mask, resample_factor_x, resample_factor_y)
    return image, mask


def resample_mask(
    mask: MaskArray, resample_factor_x: float, resample_factor_y: float
) -> MaskArray:
    """Resample the provided mask array using the specified resampling factors."""
    return resample_array(
        mask, resample_factor_x, resample_factor_y, order=0, mode="nearest"
    )


def resample_scan_image(
    image: ScanImage,
    resample_factor_x: float,
    resample_factor_y: float,
) -> ScanImage:
    """Resample the ScanImage object using the specified resampling factors."""
    image_array_resampled = resample_array(
        image.data, resample_factor_x, resample_factor_y, order=1, mode="nearest"
    )
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x / resample_factor_x,
        scale_y=image.scale_y / resample_factor_y,
    )


def resample_array(
    array: NDArray,
    resample_factor_x: float,
    resample_factor_y: float,
    order: int,
    mode: str,
) -> NDArray:
    """
    Resample an array using the specified resampling factors, order, and mode.

    :param array: The array to resample.
    :param resample_factor_x: The resampling factor for the x-axis.
    :param resample_factor_y: The resampling factor for the y-axis.
    :param order: The order of the spline interpolation to use.
    :param mode: The mode to use for handling boundaries.

    :returns: The resampled array.
    """
    resampled = ndimage.zoom(
        array,
        (resample_factor_y, resample_factor_x),
        order=order,
        mode=mode,
    )
    return np.asarray(resampled).astype(array.dtype)


def get_resampling_factors(
    scale_x: float,
    scale_y: float,
    only_downsample: bool,
    resample_factor: float | None,
    target_resolution: float,
    preserve_aspect_ratio: bool = False,
) -> tuple[float, float]:
    """
    Calculate resampling factors for x and y dimensions. If `resample_factor` is provided, factors are set to 1/resample_factor,
    otherwise, factors are calculated from current scale and target_resolution.

    The resampling factor determines how the image dimensions will change:
    - factor > 1: upsampling (more pixels, finer resolution)
    - factor < 1: downsampling (fewer pixels, coarser resolution)
    - factor = 1: no change

    :param scale_x: Scale for the x-axis
    :param scale_y: Scale for the y-axis
    :param only_downsample: If True, clamp factors to <= 1 to prevent upsampling
    :param resample_factor: Direct resampling factor. If provided, overrides target_resolution.
    :param target_resolution: Target resolution (m). Used when `resample_factor` is None.
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.

    :returns: Resampling factors.
    """
    if resample_factor is not None:
        resample_factor_x = resample_factor_y = 1 / resample_factor
    else:
        resample_factor_x = scale_x / target_resolution
        resample_factor_y = scale_y / target_resolution

    if preserve_aspect_ratio:
        # Scale both factors equally to preserve the aspect ratio
        resample_factor_x = resample_factor_y = min(
            resample_factor_x, resample_factor_y
        )

    if only_downsample:
        # if only downsampling is allowed, clip the factors at max 1
        resample_factor_x = min(1.0, resample_factor_x)
        resample_factor_y = min(1.0, resample_factor_y)
    return resample_factor_x, resample_factor_y
