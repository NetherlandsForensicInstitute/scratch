from typing import Optional

import numpy as np
from numpydantic import NDArray
from skimage.transform import resize

from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def resample_scan_image_and_mask(
    image: ScanImage,
    mask: Optional[MaskArray] = None,
    resampling_factors: Optional[tuple[float, float]] = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
) -> tuple[ScanImage, Optional[MaskArray]]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser
    than the target scale, no resampling is performed. If `resample_factors` are
    provided, it overrides the target scale.

    The resampling factor determines how the image dimensions will change:
    - factor < 1: upsampling (more pixels, finer resolution)
    - factor > 1: downsampling (fewer pixels, coarser resolution)
    - factor = 1: no change

    :param image: Input ScanImage to resample
    :param mask: Corresponding mask array
    :param resampling_factors: Resampling factors
    :param target_scale: Target scale (m) when resample_factors are not provided
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data

    :returns: Resampled ScanImage and MaskArray
    """
    if not resampling_factors:
        resampling_factors = get_resampling_factors(
            image.scale_x,
            image.scale_y,
            target_scale,
        )
    if only_downsample:
        resampling_factors = clip_resample_factors(
            resampling_factors, preserve_aspect_ratio
        )
    if resampling_factors == (1, 1):
        return image, mask

    image = resample_scan_image(image, resampling_factors)
    if mask is not None:
        mask = _resample_array(mask, resampling_factors)
    return image, mask


def resample_scan_image(
    image: ScanImage, resampling_factors: tuple[float, float]
) -> ScanImage:
    """Resample the ScanImage object using the specified resampling factors."""
    image_array_resampled = _resample_array(image.data, resampling_factors)
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x * resampling_factors[0],
        scale_y=image.scale_y * resampling_factors[1],
    )


def _resample_array(
    array: NDArray,
    resample_factors: tuple[float, float],
) -> NDArray:
    """
    Resample an array using the specified resampling factors, order, and mode.

    :param array: The array to resample.
    :param resample_factors: The resampling factors for the x- and y-axis.

    :returns: The resampled array.
    """
    # Rescale array. We do not need the "order" argument here since `skimage` sets it based on image dtype
    resampled = resize(
        image=array,
        output_shape=tuple(
            1 / factor * array.shape[i] for i, factor in enumerate(resample_factors)
        ),
        mode="edge",
        anti_aliasing=array.dtype != np.bool_
        and any(factor > 1 for factor in resample_factors),
    )
    return np.asarray(resampled, dtype=array.dtype)


def get_resampling_factors(
    scale_x: float,
    scale_y: float,
    target_scale: float,
) -> tuple[float, float]:
    """
    Calculate resampling factors for x and y dimensions.

    :param scale_x: Scale for the x-axis
    :param scale_y: Scale for the y-axis
    :param target_scale: Target pixel size (in meters).

    :returns: Resampling factors.
    """
    resampling_factor_x = target_scale / scale_x
    resampling_factor_y = target_scale / scale_y
    return resampling_factor_x, resampling_factor_y


def clip_resample_factors(
    resampling_factors: tuple[float, float],
    preserve_aspect_ratio: bool,
) -> tuple[float, float]:
    """Clip the resampling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        # Scale both factors equally to preserve the aspect ratio
        max_factor = max(resampling_factors)
        resampling_factors = (max_factor, max_factor)

    resampling_factors = (
        max(resampling_factors[0], 1.0),
        max(resampling_factors[1], 1.0),
    )
    return resampling_factors
