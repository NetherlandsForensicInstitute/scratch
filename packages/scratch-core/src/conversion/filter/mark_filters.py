"""
Mark-specific filtering functions and pipelines.

This module provides filtering operations for Mark objects, including
anti-aliasing and filter pipelines for preprocessing.
"""

from conversion.data_formats import Mark
from conversion.filter.gaussian import apply_gaussian_regression_filter
from conversion.preprocess_impression.utils import update_mark_data
from conversion.resample import get_scaling_factors


def apply_gaussian_filter_mark(
    mark: Mark,
    cutoff: float,
    regression_order: int,
    is_high_pass: bool,
) -> Mark:
    """
    Apply 2D Gaussian filter to mark data.

    :param mark: Input mark.
    :param cutoff: Filter cutoff length.
    :param regression_order: Order of the local polynomial fit (0, 1, or 2).
    :param is_high_pass: If True, apply high-pass filter; otherwise low-pass.
    :return: Filtered mark.
    """

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=is_high_pass,
        cutoff_length=cutoff,
        regression_order=regression_order,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
    )
    return update_mark_data(mark, filtered_data)


def apply_filter_pipeline(
    mark: Mark,
    target_scale: float | None,
    lowpass_cutoff: float | None,
    lowpass_regression_order: int,
) -> tuple[Mark, Mark, float | None]:
    """
    Apply the filtering pipeline to a leveled mark: anti-aliasing and low-pass filtering.

    Anti-aliasing is implemented using a zero-order Gaussian regression filter, which effectively
    acts as a low-pass filter to suppress frequencies above the Nyquist limit when resampling.

    :param mark: Leveled mark.
    :param target_scale: Target pixel scale in meters for resampling (None to skip anti-aliasing)
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    :return: Tuple of (filtered mark, anti-aliased-only mark, anti-alias cutoff).
    """
    if target_scale is None:
        mark_anti_aliased, anti_alias_cutoff = mark, None
    else:
        mark_anti_aliased, anti_alias_cutoff = _apply_anti_aliasing(mark, target_scale)

    # Only apply an additional low-pass filter if `lowpass_cutoff` is defined and is bigger than the `anti_alias_cutoff`
    if lowpass_cutoff is not None and (
        anti_alias_cutoff is None or lowpass_cutoff < anti_alias_cutoff
    ):
        mark_filtered = apply_gaussian_filter_mark(
            mark,
            lowpass_cutoff,
            lowpass_regression_order,
            is_high_pass=False,
        )
    else:
        mark_filtered = mark_anti_aliased

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff


def _apply_anti_aliasing(
    mark: Mark,
    target_scale: float,
) -> tuple[Mark, float | None]:
    """
    Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from aliasing when
    resampling to a coarser resolution. Applied when downsampling by >1.5x.

    :param mark: Input mark.
    :param target_scale: Target scale in meters.
    :return: Tuple of (filtered mark, cutoff wavelength applied).
    """
    factors = get_scaling_factors(
        scales=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        target_scale=target_scale,
    )

    # Only filter if downsampling by >1.5x
    if all(r <= 1.5 for r in factors):
        return mark, None

    filtered_data = apply_gaussian_regression_filter(
        mark.scan_image.data,
        is_high_pass=False,
        regression_order=0,
        pixel_size=(mark.scan_image.scale_x, mark.scan_image.scale_y),
        cutoff_length=target_scale,
    )
    return update_mark_data(mark, filtered_data), target_scale
