from typing import Optional

import numpy as np
from numpydantic import NDArray
from skimage.transform import resize

from image_generation.data_formats import ScanImage
from utils.array_definitions import MaskArray


def resample_image_and_mask(
    scan_image: ScanImage,
    mask: Optional[MaskArray] = None,
    factors: Optional[tuple[float, float]] = None,
    target_scale: float = 4e-6,
    only_downsample: bool = True,
    preserve_aspect_ratio: bool = True,
) -> tuple[ScanImage, Optional[MaskArray]]:
    """
    Resample the input image and optionally its corresponding mask.

    If `only_downsample` is True and the current resolution is already coarser
    than the target scale, no resampling is performed. If `scale_multipliers` are
    provided, it overrides the target scale.

    :param scan_image: Input ScanImage to resample.
    :param mask: Corresponding mask array.
    :param factors: The scaling multipliers for the X- and Y-axis. The formula used is `new_scale = factor * old_scale`.
    :param target_scale: Target scale (in meters) when `factors` are not provided.
    :param preserve_aspect_ratio: Whether to preserve the aspect ratio of the image.
    :param only_downsample: If True, only downsample data (default). If False, no rescaling is performed.
    :returns: Resampled ScanImage and MaskArray
    """
    if not factors:
        factors = get_scaling_factors(
            scales=(scan_image.scale_x, scan_image.scale_y), target_scale=target_scale
        )
    if only_downsample:
        factors = clip_factors(factors, preserve_aspect_ratio)
    factor_x, factor_y = factors

    if np.allclose(factors, 1.0):
        return scan_image, mask

    image = resample_scan_image(scan_image, factor_x=factor_x, factor_y=factor_y)
    if mask is not None:
        mask = _resample_array(mask, factors=(factor_y, factor_x))
    return image, mask


def resample_scan_image(
    image: ScanImage, factor_x: float, factor_y: float
) -> ScanImage:
    """Resample the ScanImage object using the specified resampling factors."""
    image_array_resampled = _resample_array(image.data, factors=(factor_y, factor_x))
    return ScanImage(
        data=image_array_resampled,
        scale_x=image.scale_x * factor_x,
        scale_y=image.scale_y * factor_y,
    )


def _resample_array(
    array: NDArray,
    factors: tuple[float, ...],
) -> NDArray:
    """
    Resample an array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    :param array: The array to resample.
    :param factors: The resampling factors per axis.\
    :returns: The resampled array.
    """
    # Rescale array. We do not need the "order" argument here since `skimage` sets it based on image dtype
    resampled = resize(
        image=array,
        output_shape=tuple(
            1 / factor * array.shape[i] for i, factor in enumerate(factors)
        ),
        mode="edge",
        anti_aliasing=any(factor > 1 for factor in factors),
    ).astype(array.dtype)
    return resampled


def get_scaling_factors(
    scales: tuple[float, ...],
    target_scale: float,
) -> tuple[float, ...]:
    """
    Calculate the scaling multipliers for a target scale.

    :param scales: Current scales (= pixel size in meters per axis).
    :param target_scale: Target scale (= pixel size in meters).

    :returns: The computed scaling multipliers.
    """
    return tuple(target_scale / scale for scale in scales)


def clip_factors(
    factors: tuple[float, ...],
    preserve_aspect_ratio: bool,
) -> tuple[float, ...]:
    """Clip the scaling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        # Set the multipliers to equal values to preserve the aspect ratio
        max_factor = max(factors)
        factors = tuple(max_factor for _ in factors)

    clipped = tuple(max(factor, 1.0) for factor in factors)
    return clipped
