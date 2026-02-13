"""Preprocessing pipeline for impression mark scan images.

This module provides functions to preprocess 2D scan images of impression marks
(e.g., breech face impressions) through leveling, filtering, and resampling steps.
"""

from container_models.base import DepthData
from conversion.data_formats import Mark
from conversion.filter import (
    apply_gaussian_filter_mark,
    apply_filter_pipeline,
)
from conversion.leveling import SurfaceTerms, level_map
from conversion.mask import crop_to_mask
from conversion.preprocess_impression.center import compute_center_local
from conversion.preprocess_impression.resample import (
    resample,
    needs_resampling,
)
from conversion.preprocess_impression.tilt import apply_tilt_correction
from conversion.preprocess_impression.utils import update_mark_data, Point2D
from conversion.resample import get_scaling_factors, resample_array_2d
from preprocessors.schemas import PreprocessingImpressionParams


def preprocess_impression_mark(
    mark: Mark,
    params: PreprocessingImpressionParams,
) -> tuple[Mark, Mark]:
    """
    Preprocess trimmed impression image data.

    Processing pipeline:

    1. Compute image center and crop NaN borders
    2. Remove tilt from image (optional)
    3. Level data with configured surface terms (either with mean, mean+tilt or mean+tilt+quadratic curve)
    4. Apply antialiasing filter (if downsampling)
    5. Apply low-pass filter (if stronger than antialiasing) to avoid resolution artifacts when downsampling
    6. Resample to target resolution
    7. Apply high-pass filter
    8. Final leveling pass (same as 3.)

    :param mark: Input mark with trimmed impression data.
    :param params: Processing parameters.
    :return: Tuple of (filtered mark, leveled-only mark).
    """
    # Stage 1: Preparation
    mark = _prepare_mark(mark)

    # Stage 2: Tilt correction
    if params.adjust_pixel_spacing:
        mark = apply_tilt_correction(mark)

    # Stage 3: Initial leveling
    mark_leveled, fitted_surface = _level_mark(mark, params.surface_terms)

    # Stage 4-5: Filtering (antialiasing + low-pass)
    mark_filtered, mark_anti_aliased, anti_alias_cutoff = apply_filter_pipeline(
        mark_leveled,
        target_scale=params.pixel_size,
        lowpass_cutoff=params.lowpass_cutoff,
        lowpass_regression_order=params.lowpass_regression_order,
    )

    # Stage 6: Resampling
    if params.pixel_size is not None:
        original_scales = (
            mark_filtered.scan_image.scale_x,
            mark_filtered.scan_image.scale_y,
        )
        if needs_resampling(mark_filtered, params.pixel_size):
            mark_filtered = resample(mark_filtered, params.pixel_size)
            mark_anti_aliased = resample(mark_anti_aliased, params.pixel_size)
            factors = get_scaling_factors(
                scales=original_scales, target_scale=params.pixel_size
            )
            fitted_surface = resample_array_2d(fitted_surface, factors=factors)

    # Stage 7: High-pass filter
    if params.highpass_cutoff is not None:
        mark_filtered = apply_gaussian_filter_mark(
            mark_filtered,
            params.highpass_cutoff,
            params.highpass_regression_order,
            is_high_pass=True,
        )

    # Stage 8: Final leveling
    mark_filtered, _ = _level_mark(mark_filtered, params.surface_terms, mark.center)

    # Prepare leveled-only output
    mark_leveled_final = _finalize_leveled_output(
        mark_anti_aliased,
        fitted_surface,
        params.pixel_size,
        params.surface_terms,
        mark.center,
    )

    # Build output metadata
    mark.meta_data.update(**params.dict())

    return mark_filtered, mark_leveled_final


def _level_mark(
    mark: Mark,
    terms: SurfaceTerms,
    reference_point: Point2D | None = None,
) -> tuple[Mark, DepthData]:
    result = level_map(
        mark.scan_image, terms=terms, reference_point=reference_point or mark.center
    )
    leveled_mark = update_mark_data(mark, result.leveled_map)
    return leveled_mark, result.fitted_surface


def _prepare_mark(mark: Mark) -> Mark:
    """
    Initial preparation: compute center and crop NaN borders.

    :param mark: Input mark.
    :return: Cropped mark
    """
    center_local = compute_center_local(mark)
    cropped_data = crop_to_mask(mark.scan_image.data, mark.scan_image.valid_mask)
    return update_mark_data(mark, data=cropped_data, center=center_local)


def _finalize_leveled_output(
    mark: Mark,
    fitted_surface: DepthData,
    target_scale: float | None,
    surface_terms: SurfaceTerms,
    reference_point: Point2D,
) -> Mark:
    """
    Prepare the leveled-only output.

    :param mark: Mark after tilt correction, before SPHERE leveling.
    :param target_scale: Target pixel scale for resampling, or None to skip.
    :param surface_terms: Original surface terms (will be masked to PLANE).
    :param reference_point: Reference point for leveling.
    :return: Final leveled mark.
    """
    # Add back fitted surface (restores curvature)
    restored_data = mark.scan_image.data + fitted_surface
    mark_restored = update_mark_data(mark, restored_data)

    # Resample if needed
    if target_scale is not None:
        mark_restored = resample(mark_restored, target_scale)

    # Apply PLANE-only leveling (after resampling, like MATLAB)
    rigid_terms = surface_terms & SurfaceTerms.PLANE
    leveled_mark, _ = _level_mark(mark_restored, rigid_terms, reference_point)

    return leveled_mark
