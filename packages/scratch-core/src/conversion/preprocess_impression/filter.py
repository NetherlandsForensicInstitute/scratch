from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.filter import apply_gaussian_regression_filter
from conversion.preprocess_impression.utils import (
    update_mark_data,
    _update_mark_scan_image,
)
from conversion.resample import _get_scaling_factors


def _apply_anti_aliasing(
    scan_image: ScanImage,
    target_scale: float,
) -> tuple[ScanImage, float | None]:
    """
    Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from aliasing when
    resampling to a coarser resolution. Applied when downsampling by >1.5x.

    :param scan_image: Input scan image.
    :param target_scale: Target scale (x, y) in meters.
    :return: Tuple of (filtered image, cutoff wavelengths applied).
    """
    factors = _get_scaling_factors(
        scales=(scan_image.scale_x, scan_image.scale_y), target_scale=target_scale
    )

    # Only filter if downsampling by >1.5x
    if all(r <= 1.5 for r in factors):
        return scan_image, None

    cutoff = target_scale

    filtered_data = apply_gaussian_regression_filter(
        scan_image.data,
        is_high_pass=False,
        regression_order=0,
        pixel_size=(scan_image.scale_x, scan_image.scale_y),
        cutoff_length=cutoff,
    )

    return scan_image.model_copy(update={"data": filtered_data}), cutoff


def apply_gaussian_filter_to_mark(
    mark: Mark,
    cutoff: float,
    regression_order: int,
    *,
    is_high_pass: bool,
) -> Mark:
    """
    Apply Gaussian filter to mark data.

    :param mark: Input mark.
    :param cutoff: Filter cutoff length.
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


def apply_filtering_pipeline(
    mark: Mark,
    pixel_size: float | None,
    lowpass_cutoff: float | None,
    lowpass_regression_order: int,
) -> tuple[Mark, Mark, float | None]:
    """
    Apply the filtering pipeline: anti-aliasing (which is basically a zero-order low-pass filter) and low-pass (as
    additional filter when the lowpass cutoff is larger than the anti-aliasing cutoff) .

    :param mark: Leveled mark.
    :param pixel_size: Target pixel spacing in meters for resampling
    :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
    :param lowpass_regression_order: Order of the local polynomial fit (0, 1, or 2) in low pass filters.
    :return: Tuple of (filtered mark, anti-aliased-only mark, anti-alias cutoffs).
    """
    if pixel_size is not None:
        anti_aliased_scan, anti_alias_cutoff = _apply_anti_aliasing(
            mark.scan_image, pixel_size
        )
        mark_anti_aliased = _update_mark_scan_image(mark, anti_aliased_scan)
    else:
        mark_anti_aliased = mark
        anti_alias_cutoff = None

    # Low-pass decision (for map path)
    if lowpass_cutoff is None:
        # No low-pass configured, use anti-aliased
        mark_filtered = mark_anti_aliased
    else:
        if anti_alias_cutoff and lowpass_cutoff >= anti_alias_cutoff:
            # Anti-aliasing is sufficient
            mark_filtered = mark_anti_aliased
        else:
            # Apply low-pass to original leveled (not anti-aliased)
            mark_filtered = apply_gaussian_filter_to_mark(
                mark,
                lowpass_cutoff,
                lowpass_regression_order,
                is_high_pass=False,
            )

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff
