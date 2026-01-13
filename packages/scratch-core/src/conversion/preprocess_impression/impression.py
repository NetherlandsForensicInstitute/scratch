"""Preprocessing pipeline for impression mark scan images.

This module provides functions to preprocess 2D scan images of impression marks
(e.g., breech face impressions) through leveling, filtering, and resampling steps.
"""

from dataclasses import asdict

from container_models.base import ScanMap2DArray
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark
from conversion.leveling import SurfaceTerms, level_map
from conversion.preprocess_impression.center import compute_center_local
from conversion.preprocess_impression.crop import crop_nan_borders
from conversion.preprocess_impression.filter import (
    apply_gaussian_filter_to_mark,
    apply_filtering_pipeline,
)
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_impression.resample import (
    resample,
    apply_resampling_pipeline,
)
from conversion.preprocess_impression.tilt import apply_tilt_correction
from conversion.preprocess_impression.utils import update_mark_data, Point2D


def _level_mark(
    mark: Mark,
    terms: SurfaceTerms,
    reference_point: Point2D | None = None,
) -> tuple[Mark, ScanMap2DArray]:
    result = level_map(mark.scan_image, terms=terms, reference_point=reference_point)
    leveled_mark = update_mark_data(mark, result.leveled_map)
    return leveled_mark, result.fitted_surface


def _build_output_mark(
    mark: Mark,
    output_scale: tuple[float, float],
    *,
    is_filtered: bool,
) -> Mark:
    """
    Construct output Mark with proper scale and  metadata.

    :param mark: Source mark.
    :param output_scale: Output scale(x, y).
    :param is_filtered: Whether filtering was applied.
    :return: New Mark with updated scale and metadata.
    """
    scan_image = ScanImage(
        data=mark.scan_image.data,
        scale_x=output_scale[0],
        scale_y=output_scale[1],
    )
    return Mark(
        scan_image=scan_image,
        mark_type=mark.mark_type,
        crop_type=mark.crop_type,
        meta_data=mark.meta_data | {"is_filtered": is_filtered, "is_leveled": True},
    )


def _build_preprocessing_metadata(
    params: PreprocessingImpressionParams,
    center_local: Point2D,
    interpolated: bool,
) -> dict:
    """
    Build metadata dictionary for preprocessed mark.

    :param params: Preprocessing parameters.
    :param center_local: Local center coordinates.
    :param interpolated: Whether interpolation was performed.
    :return: Metadata dictionary.
    """
    return {
        **asdict(params),
        "center_g_x": 0,
        "center_g_y": 0,
        "center_l_x": center_local[0],
        "center_l_y": center_local[1],
        "is_crop": True,
        "is_prep": True,
        "is_interpolated": interpolated,
    }


def _prepare_mark(mark: Mark) -> tuple[Mark, Point2D]:
    """
    Initial preparation: compute center and crop NaN borders.

    :param mark: Input mark.
    :return: Tuple of (cropped mark, local center in meters).
    """
    center_local = compute_center_local(mark)
    cropped_data = crop_nan_borders(mark.scan_image.data)
    return update_mark_data(mark, cropped_data), center_local


def _finalize_leveled_output(
    mark: Mark,
    fitted_surface: ScanMap2DArray,
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
        mark_restored, _ = resample(mark_restored, target_scale)

    # Apply PLANE-only leveling (after resampling, like MATLAB)
    rigid_terms = surface_terms & SurfaceTerms.PLANE
    leveled_mark, _ = _level_mark(mark_restored, rigid_terms, reference_point)

    return leveled_mark


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
    mark, center_local = _prepare_mark(mark)

    # Stage 2: Tilt correction
    mark, center_local = apply_tilt_correction(
        mark, center_local, params.adjust_pixel_spacing
    )

    # Stage 3: Initial leveling
    mark_leveled, fitted_surface = _level_mark(mark, params.surface_terms, center_local)

    # Stage 4-5: Filtering (antialiasing + low-pass)
    mark_filtered, mark_anti_aliased, anti_alias_cutoff = apply_filtering_pipeline(
        mark_leveled,
        pixel_size=params.pixel_size,
        lowpass_cutoff=params.lowpass_cutoff,
        lowpass_regression_order=params.lowpass_regression_order,
    )

    # Stage 6: Resampling
    interpolated = False
    if params.pixel_size is not None:
        mark_filtered, mark_anti_aliased, fitted_surface, interpolated = (
            apply_resampling_pipeline(
                mark_filtered, mark_anti_aliased, fitted_surface, params.pixel_size
            )
        )

    # Stage 7: High-pass filter
    if params.highpass_cutoff is not None:
        mark_filtered = apply_gaussian_filter_to_mark(
            mark_filtered,
            params.highpass_cutoff,
            params.highpass_regression_order,
            is_high_pass=True,
        )

    # Stage 8: Final leveling
    mark_filtered, _ = _level_mark(mark_filtered, params.surface_terms, center_local)

    # Prepare leveled-only output
    mark_leveled_final = _finalize_leveled_output(
        mark_anti_aliased,
        fitted_surface,
        params.pixel_size,
        params.surface_terms,
        center_local,
    )

    # Build output metadata
    mark.meta_data.update(
        _build_preprocessing_metadata(params, center_local, interpolated)
    )

    output_pixel_size = (
        (params.pixel_size, params.pixel_size)
        if params.pixel_size
        else (
            mark.scan_image.scale_x,
            mark.scan_image.scale_y,
        )
    )
    return (
        _build_output_mark(mark_filtered, output_pixel_size, is_filtered=True),
        _build_output_mark(mark_leveled_final, output_pixel_size, is_filtered=False),
    )
