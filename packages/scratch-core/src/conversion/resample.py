from typing import Optional

import numpy as np
from scipy import ndimage

from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def resample(
    image: ScanImage,
    mask: Optional[MaskArray] = None,
    sampling: Optional[float] = None,
    target_sampling_distance: float = 4e-6,
    only_downsample: bool = True,
) -> tuple[ScanImage, Optional[MaskArray]]:
    """
    Resample image (and mask if provided) data.

    If `only_downsample` and the current resolution is already coarser than the target,
    no resampling is performed.

    :param image: Input ScanImage
    :param mask: Mask array
    :param sampling: Resampling factor (1/sampling is applied). If None, resamples to target_pixelsize.
    :param target_sampling_distance: Target sampling distance (m) when sampling is not provided
    :param only_downsample: If True, only downsample data

    :returns: Resampled ScanImage and MaskArray
    """
    resample_factor_x, resample_factor_y = get_resampling_factors(
        image.scale_x,
        image.scale_y,
        only_downsample,
        sampling,
        target_sampling_distance,
    )

    if resample_factor_x == 1 and resample_factor_y == 1:
        return image, mask

    image_array_resampled = ndimage.zoom(
        image.data,
        (resample_factor_y, resample_factor_x),
        order=1,  # bilinear
        mode="nearest",
    )
    image_array_resampled = np.asarray(image_array_resampled, dtype=image.data.dtype)
    if mask is not None:
        mask_resampled = ndimage.zoom(
            mask,
            (resample_factor_y, resample_factor_x),
            order=0,  # nearest
            mode="nearest",
        )
        mask = np.asarray(mask_resampled, dtype=mask.dtype)
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x / resample_factor_x,
        scale_y=image.scale_y / resample_factor_y,
    ), mask


def get_resampling_factors(
    scale_x: float,
    scale_y: float,
    only_downsample: bool,
    sampling: float | None,
    target_sampling_distance: float,
) -> tuple[float, float]:
    """
    Calculate resampling factors for x and y dimensions. If `sampling` is provided, factors are set to 1/sampling,
    otherwise, factors are calculated from current scale and target_sampling_distance.

    The resampling factor determines how the image dimensions will change:
    - factor > 1: upsampling (more pixels, finer resolution)
    - factor < 1: downsampling (fewer pixels, coarser resolution)
    - factor = 1: no change

    :param scale_x: Scale for the x-axis
    :param scale_y: Scale for the y-axis
    :param only_downsample: If True, clamp factors to <= 1 to prevent upsampling
    :param sampling: Direct resampling factor. If provided, overrides target_sampling_distance.
    :param target_sampling_distance: Target sampling distance [m]. Used when sampling is None.

    :returns: Resampling factors for (x, y) dimensions.
    """
    if sampling is not None:
        resample_factor_x = resample_factor_y = 1 / sampling
    else:
        resample_factor_x = scale_x / target_sampling_distance
        resample_factor_y = scale_y / target_sampling_distance

    if only_downsample:
        # if only downsampling is allowed, clip the factors at max 1
        resample_factor_x = min(1.0, resample_factor_x)
        resample_factor_y = min(1.0, resample_factor_y)
    return resample_factor_x, resample_factor_y
