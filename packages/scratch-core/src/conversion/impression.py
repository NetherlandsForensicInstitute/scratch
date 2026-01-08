"""Preprocessing pipeline for impression mark scan images.

This module provides functions to preprocess 2D scan images of impression marks
(e.g., breech face impressions) through leveling, filtering, and resampling steps.
"""

from dataclasses import asdict
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion
from skimage.measure import CircleModel, ransac

from container_models.base import MaskArray, ScanMap2DArray
from container_models.scan_image import ScanImage
from conversion.crop import crop_nan_borders
from conversion.data_formats import Mark, MarkType
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling import SurfaceTerms, level_map
from conversion.parameters import PreprocessingImpressionParams
from conversion.resample import resample_scan_image_and_mask

# Type aliases
Point2D = tuple[float, float]
FilterCutoff = tuple[float | None, float | None]


class TiltEstimate(NamedTuple):
    """Result of plane tilt estimation."""

    tilt_x_deg: float
    tilt_y_deg: float
    residuals: NDArray[np.floating]


def _update_mark_data(mark: Mark, data: NDArray) -> Mark:
    """
    Return a new Mark with updated scan data.

    :param mark: Original mark.
    :param data: New data array.
    :return: New Mark instance with updated data.
    """
    scan_image = mark.scan_image.model_copy(update={"data": data})
    return mark.model_copy(update={"scan_image": scan_image})


def _update_mark_scan_image(mark: Mark, scan_image: ScanImage) -> Mark:
    """
    Return a new Mark with updated scan image.

    :param mark: Original mark.
    :param scan_image: New scan image.
    :return: New Mark instance with updated scan image.
    """
    return mark.model_copy(update={"scan_image": scan_image})


def _get_mask_edge_points(mask: MaskArray) -> NDArray[np.floating]:
    """
    Extract inner edge points of a binary mask.

    :param mask: Binary mask array.
    :return: Array of (col, row) edge points in pixel coordinates.
    """
    eroded = binary_erosion(mask).astype(bool)
    edge = mask & ~eroded
    rows, cols = np.where(edge)
    return np.column_stack([cols, rows]).astype(float)


def _points_are_collinear(points: NDArray[np.floating], tol: float = 1e-9) -> bool:
    """
    Check if points are approximately collinear using SVD.

    :param points: Array of 2D points.
    :param tol: Tolerance for collinearity check.
    :return: True if points are collinear.
    """
    if len(points) < 3:
        return True
    centered = points - points.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    return singular_values[-1] < tol * singular_values[0]


def _fit_circle_ransac(
    points: NDArray[np.floating],
    n_iterations: int = 1000,
    threshold: float = 1.0,
) -> Point2D | None:
    """
    Fit a circle to 2D points using RANSAC.

    :param points: Array of (x, y) points with shape (N, 2).
    :param n_iterations: Maximum RANSAC iterations.
    :param threshold: Inlier distance threshold in same units as points.
    :return: Circle center (x, y) or None if fitting failed.
    :raises ValueError: If points array has wrong shape.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got {points.shape}")

    if _points_are_collinear(points):
        return None

    model, _ = ransac(
        points,
        CircleModel,
        min_samples=3,
        residual_threshold=threshold,
        max_trials=n_iterations,
    )

    if model is None:
        return None

    x, y, radius = model.params
    if radius > 0 and np.isfinite([x, y, radius]).all():
        return x, y

    return None


def _get_bounding_box_center(mask: NDArray[np.bool_]) -> Point2D:
    """
    Compute center of bounding box for True values in mask.

    :param mask: Boolean mask array.
    :return: Center (x, y) in pixel coordinates.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return mask.shape[1] / 2, mask.shape[0] / 2
    return (
        (cols.min() + cols.max() + 1) / 2,
        (rows.min() + rows.max() + 1) / 2,
    )


def _compute_map_center(
    data: ScanMap2DArray,
    use_circle_fit: bool = False,
) -> Point2D:
    """
    Compute map center from data bounds or circle fit.

    :param data: Height map array.
    :param use_circle_fit: If True, attempt RANSAC circle fitting first.
    :return: Center position (col, row) in pixel coordinates.
    """
    valid_mask = ~np.isnan(data)

    if use_circle_fit:
        edge_points = _get_mask_edge_points(valid_mask)
        if (center := _fit_circle_ransac(edge_points)) is not None:
            return center

    return _get_bounding_box_center(valid_mask)


def _estimate_plane_tilt(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> TiltEstimate:
    """
    Estimate best-fit plane tilt angles using least squares.

    Fits z = ax + by + c to the data points.

    :param x: X coordinates.
    :param y: Y coordinates.
    :param z: Z values at each (x, y).
    :return: TiltEstimate with tilt angles in degrees and residuals.
    """
    design_matrix = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(design_matrix, z, rcond=None)

    return TiltEstimate(
        tilt_x_deg=np.degrees(np.arctan(a)),
        tilt_y_deg=np.degrees(np.arctan(b)),
        residuals=z - (a * x + b * y + c),
    )


def _get_valid_coordinates(
    scan_image: ScanImage,
    center: Point2D,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract x, y, z coordinates of valid pixels, centered at origin.

    :param scan_image: Input scan image.
    :param center: Center point to subtract (in pixel coordinates).
    :return: Tuple of (x, y, z) coordinate arrays in meters.
    """
    valid_mask = ~np.isnan(scan_image.data)
    rows, cols = np.where(valid_mask)

    x = cols * scan_image.scale_x - center[0]
    y = rows * scan_image.scale_y - center[1]
    z = scan_image.data[valid_mask]

    return x, y, z


def _adjust_for_plane_tilt(
    scan_image: ScanImage,
    center: Point2D,
) -> tuple[ScanImage, Point2D]:
    """
    Remove plane tilt from scan image and adjust scale factors.

    :param scan_image: Input scan image.
    :param center: Center position in meters.
    :return: Tuple of (leveled scan image, adjusted center).
    :raises ValueError: If fewer than 3 valid points exist.
    """
    x, y, z = _get_valid_coordinates(scan_image, center)

    if len(x) < 3:
        raise ValueError("Need at least 3 valid points to estimate plane tilt")

    tilt = _estimate_plane_tilt(x, y, z)

    # Update data with residuals
    data = scan_image.data.copy()
    data[~np.isnan(scan_image.data)] = tilt.residuals

    # Adjust scales for tilt
    cos_x = np.cos(np.radians(tilt.tilt_x_deg))
    cos_y = np.cos(np.radians(tilt.tilt_y_deg))
    scale_x_new = scan_image.scale_x / cos_x
    scale_y_new = scan_image.scale_y / cos_y

    # Adjust center for new scales
    center_new = (
        center[0] * scale_x_new / scan_image.scale_x,
        center[1] * scale_y_new / scan_image.scale_y,
    )

    return ScanImage(data=data, scale_x=scale_x_new, scale_y=scale_y_new), center_new


def _compute_center_local(mark: Mark) -> Point2D:
    """
    Compute local center coordinates from mark data.

    :param mark: Input mark image.
    :return: Center (x, y) in meters.
    """
    use_circle = mark.mark_type == MarkType.BREECH_FACE_IMPRESSION
    center_px = _compute_map_center(mark.scan_image.data, use_circle_fit=use_circle)
    return (
        center_px[0] * mark.scan_image.scale_x,
        center_px[1] * mark.scan_image.scale_y,
    )


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


def _apply_gaussian_filter_to_mark(
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
    return _update_mark_data(mark, filtered_data)


def _level_mark(
    mark: Mark,
    terms: SurfaceTerms,
    reference_point: Point2D | None = None,
) -> Mark:
    result = level_map(mark.scan_image, terms=terms, reference_point=reference_point)
    leveled_mark = _update_mark_data(mark, result.leveled_map)
    return leveled_mark


def _needs_resampling(mark: Mark, target_scale: float) -> bool:
    """
    Check if mark needs resampling to target scale.

    :param mark: Input mark.
    :param target_scale: Target pixel scale.
    :return: True if resampling is needed.
    """
    current_scale = (mark.scan_image.scale_x, mark.scan_image.scale_y)
    return not np.allclose(current_scale, (target_scale, target_scale), rtol=1e-7)


def _resample(mark: Mark, target_scale: float | None) -> tuple[Mark, bool]:
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


def _build_output_mark(
    mark: Mark,
    pixel_size: Point2D,
    *,
    is_filtered: bool,
) -> Mark:
    """
    Construct output Mark with proper scale and metadata.

    :param mark: Source mark.
    :param pixel_size: Output pixel size (x, y).
    :param is_filtered: Whether filtering was applied.
    :return: New Mark with updated scale and metadata.
    """
    scan_image = ScanImage(
        data=mark.scan_image.data,
        scale_x=pixel_size[0],
        scale_y=pixel_size[1],
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
    center_local = _compute_center_local(mark)
    cropped_data = crop_nan_borders(mark.scan_image.data)
    return _update_mark_data(mark, cropped_data), center_local


def _apply_tilt_correction(
    mark: Mark,
    center_local: Point2D,
    should_adjust: bool,
) -> tuple[Mark, Point2D]:
    """
    Apply tilt correction if requested.

    :param mark: Input mark.
    :param center_local: Local center coordinates.
    :param should_adjust: Whether to apply tilt correction.
    :return: Tuple of (corrected mark, updated center).
    """
    if not should_adjust:
        return mark, center_local

    adjusted_scan, center_local = _adjust_for_plane_tilt(mark.scan_image, center_local)
    return _update_mark_scan_image(mark, adjusted_scan), center_local


def _apply_filtering_pipeline(
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
            mark_filtered = _apply_gaussian_filter_to_mark(
                mark,
                params.lowpass_cutoff,
                params.regression_order_low,
                is_high_pass=False,
            )

    return mark_filtered, mark_anti_aliased, anti_alias_cutoff


def _apply_resampling_pipeline(
    mark_filtered: Mark,
    mark_anti_aliased: Mark,
    target_scale: float | None,
) -> tuple[Mark, Mark, bool]:
    """
    Resample marks and fitted surface to target scale.

    :param mark_filtered: Filtered mark.
    :param mark_anti_aliased: Anti-aliased-only mark.
    :param target_scale: Target pixel scale, or None to skip.
    :return: Tuple of (resampled filtered, resampled anti-aliased, resampled surface, interpolated flag).
    """
    if target_scale is None:
        return mark_filtered, mark_anti_aliased, False

    mark_filtered, filtered_resampled = _resample(mark_filtered, target_scale)
    mark_anti_aliased, anti_aliased_resampled = _resample(
        mark_anti_aliased, target_scale
    )

    interpolated = filtered_resampled or anti_aliased_resampled
    return mark_filtered, mark_anti_aliased, interpolated


def _finalize_leveled_output(
    mark_after_tilt: Mark,
    target_scale: float | None,
    surface_terms: SurfaceTerms,
    reference_point: Point2D,
) -> Mark:
    """
    Prepare the leveled-only output.

    :param mark_after_tilt: Mark after tilt correction, before SPHERE leveling.
    :param target_scale: Target pixel scale for resampling, or None to skip.
    :param surface_terms: Original surface terms (will be masked to PLANE).
    :param reference_point: Reference point for leveling.
    :return: Final leveled mark.
    """
    # Apply PLANE-only leveling (preserves curvature)
    rigid_terms = surface_terms & SurfaceTerms.PLANE
    leveled_mark = _level_mark(mark_after_tilt, rigid_terms, reference_point)

    # Resample if needed
    if target_scale is not None:
        leveled_mark, _ = _resample(leveled_mark, target_scale)

    return leveled_mark


def preprocess_impression_mark(
    mark: Mark,
    params: PreprocessingImpressionParams,
) -> tuple[Mark, Mark]:
    """
    Preprocess trimmed impression image data.

    Processing pipeline:

    1. Compute image center and crop NaN borders
    2. Adjust pixel spacing for sample tilt (optional)
    3. Level data with configured surface terms
    4. Apply anti-aliasing filter (if downsampling)
    5. Apply low-pass filter (if stronger than anti-aliasing)
    6. Resample to target resolution
    7. Apply high-pass filter
    8. Final leveling pass

    :param mark: Input mark with trimmed impression data.
    :param params: Processing parameters.
    :return: Tuple of (filtered mark, leveled-only mark).
    """
    # Stage 1: Preparation
    mark, center_local = _prepare_mark(mark)

    # Stage 2: Tilt correction
    mark, center_local = _apply_tilt_correction(
        mark, center_local, params.adjust_pixel_spacing
    )

    # Stage 3: Initial leveling
    mark_leveled = _level_mark(mark, params.surface_terms, center_local)

    # Stage 4-5: Filtering (anti-aliasing + low-pass)
    mark_filtered, mark_anti_aliased, anti_alias_cutoff = _apply_filtering_pipeline(
        mark_leveled, params
    )

    # Stage 6: Resampling
    target_scale = params.pixel_size[0] if params.pixel_size else None
    mark_filtered, mark_anti_aliased, interpolated = _apply_resampling_pipeline(
        mark_filtered, mark_anti_aliased, target_scale
    )

    # Stage 7: High-pass filter
    if params.highpass_cutoff is not None:
        mark_filtered = _apply_gaussian_filter_to_mark(
            mark_filtered,
            params.highpass_cutoff,
            params.regression_order_high,
            is_high_pass=True,
        )

    # Stage 8: Final leveling
    mark_filtered = _level_mark(mark_filtered, params.surface_terms)

    # Prepare leveled-only output
    mark_leveled_final = _finalize_leveled_output(
        mark, target_scale, params.surface_terms, center_local
    )

    # Build output metadata
    mark.meta_data.update(
        _build_preprocessing_metadata(params, center_local, interpolated)
    )

    output_pixel_size = params.pixel_size or (
        mark.scan_image.scale_x,
        mark.scan_image.scale_y,
    )
    return (
        _build_output_mark(mark_filtered, output_pixel_size, is_filtered=True),
        _build_output_mark(mark_leveled_final, output_pixel_size, is_filtered=False),
    )
