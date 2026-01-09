from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.filter import apply_gaussian_regression_filter
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_impression.utils import (
    update_mark_data,
    _update_mark_scan_image,
    Point2D,
)

FilterCutoff = tuple[float | None, float | None]


def _apply_anti_aliasing(
    scan_image: ScanImage,
    target_spacing: Point2D,
) -> tuple[ScanImage, FilterCutoff]:
    """
    Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from aliasing when
    resampling to a coarser resolution. Applied when downsampling by >1.5x.

    :param scan_image: Input scan image.
    :param target_spacing: Target pixel spacing in meters.
    :return: Tuple of (filtered image, cutoff wavelengths applied).
    """
    downsample_ratio = (
        target_spacing[0] / scan_image.scale_x,
        target_spacing[1] / scan_image.scale_y,
    )

    # Only filter if downsampling by >1.5x
    if not any(r > 1.5 for r in downsample_ratio):
        return scan_image, (None, None)

    cutoffs = (
        downsample_ratio[0] * scan_image.scale_x,
        downsample_ratio[1] * scan_image.scale_y,
    )

    filtered_data = apply_gaussian_regression_filter(
        scan_image.data,
        is_high_pass=False,
        regression_order=0,
        pixel_size=(scan_image.scale_x, scan_image.scale_y),  # todo klopt dit?
        cutoff_length=cutoffs[0],
    )

    return scan_image.model_copy(update={"data": filtered_data}), cutoffs


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
    params: PreprocessingImpressionParams,
) -> tuple[Mark, Mark, FilterCutoff]:
    """
    Apply the filtering pipeline: anti-aliasing and low-pass.

    :param mark: Leveled mark.
    :param params: Preprocessing parameters.
    :return: Tuple of (filtered mark, anti-aliased-only mark, anti-alias cutoffs).
    """
    if params.pixel_size is not None:
        anti_aliased_scan, anti_alias_cutoff = _apply_anti_aliasing(
            mark.scan_image, params.pixel_size
        )
        mark_anti_aliased = _update_mark_scan_image(mark, anti_aliased_scan)
    else:
        mark_anti_aliased = mark
        anti_alias_cutoff = (None, None)

    # Low-pass decision (for map path)
    if params.lowpass_cutoff is None:
        # No low-pass configured, use anti-aliased
        mark_filtered = mark_anti_aliased
    else:
        valid_cutoffs = [c for c in anti_alias_cutoff if c is not None]
        if valid_cutoffs and params.lowpass_cutoff >= min(valid_cutoffs):
            # Anti-aliasing is sufficient
            mark_filtered = mark_anti_aliased
        else:
            # Apply low-pass to original leveled (not anti-aliased)
            mark_filtered = apply_gaussian_filter_to_mark(
                mark,
                params.lowpass_cutoff,
                params.lowpass_regression_order,
                is_high_pass=False,
            )

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff
