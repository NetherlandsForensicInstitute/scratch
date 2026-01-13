import numpy as np

from container_models.base import ScanMap2DArray
from conversion.data_formats import Mark
from conversion.preprocess_impression.utils import _update_mark_scan_image
from conversion.resample import (
    resample_scan_image_and_mask,
    _resample_image_array,
    _get_scaling_factors,
)


def _needs_resampling(mark: Mark, target_scale: float) -> bool:
    """
    Check if mark needs resampling to target scale.

    :param mark: Input mark.
    :param target_scale: Target pixel scale.
    :return: True if resampling is needed.
    """
    current_scale = (mark.scan_image.scale_x, mark.scan_image.scale_y)
    return not np.allclose(current_scale, (target_scale, target_scale), rtol=1e-7)


def resample(mark: Mark, target_scale: float | None) -> tuple[Mark, bool]:
    """
    Resample mark if target scale differs from current scale.

    :param mark: Input mark.
    :param target_scale: Target pixel scale, or None to skip.
    :return: Tuple of (resampled mark, whether resampling occurred).
    """
    if target_scale is None or not _needs_resampling(mark, target_scale):
        return mark, False
    resampled, _ = resample_scan_image_and_mask(
        mark.scan_image,
        target_scale=target_scale,
        only_downsample=False,
    )
    return _update_mark_scan_image(mark, resampled), True


def apply_resampling_pipeline(
    mark_filtered: Mark,
    mark_anti_aliased: Mark,
    fitted_surface: ScanMap2DArray,
    target_scale: float,
) -> tuple[Mark, Mark, ScanMap2DArray, bool]:  # Fixed: 4 values
    """
    Resample marks and fitted surface to target scale.

    :param mark_filtered: Filtered mark.
    :param mark_anti_aliased: Anti-aliased-only mark.
    :param fitted_surface: Fitted surface from SPHERE leveling.
    :param target_scale: Target pixel scale, or None to skip.
    :return: Tuple of (resampled filtered, resampled anti-aliased, resampled surface, interpolated flag).
    """
    original_scales = (
        mark_anti_aliased.scan_image.scale_x,
        mark_anti_aliased.scan_image.scale_y,
    )

    mark_filtered, filtered_resampled = resample(mark_filtered, target_scale)
    mark_anti_aliased, anti_aliased_resampled = resample(
        mark_anti_aliased, target_scale
    )
    if anti_aliased_resampled:
        factors = _get_scaling_factors(
            scales=original_scales, target_scale=target_scale
        )
        fitted_surface = _resample_image_array(
            fitted_surface,
            factors=factors,
        )

    interpolated = filtered_resampled or anti_aliased_resampled
    return mark_filtered, mark_anti_aliased, fitted_surface, interpolated
