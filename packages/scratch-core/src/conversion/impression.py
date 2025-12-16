from dataclasses import asdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion
from skimage.measure import CircleModel, ransac

from conversion.crop import crop_nan_borders
from conversion.data_formats import MarkImage, MarkType
from conversion.gaussian_filter import apply_gaussian_filter
from conversion.leveling import SurfaceTerms, level_map
from conversion.parameters import PreprocessingImpressionParams
from conversion.resample import resample
from utils.array_definitions import ScanMap2DArray, MaskArray


def _get_mask_edge_points(mask: MaskArray) -> NDArray[np.floating]:
    """
    Get inner edge points of a binary mask in pixel coordinates.

    :param mask: Binary mask array
    :return: Array of (col, row) edge points in pixel indices
    """
    eroded = binary_erosion(mask).astype(bool)
    edge = mask & ~eroded

    rows, cols = np.where(edge)
    return np.column_stack([cols, rows]).astype(float)


def _points_are_collinear(points: NDArray[np.floating], tol: float = 1e-9) -> bool:
    """Check if points are approximately collinear."""
    if len(points) < 3:
        return True
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return s[-1] < tol * s[0]


def _fit_circle_ransac(
    points: NDArray[np.floating],
    n_iterations: int = 1000,
    threshold: float = 1.0,
) -> tuple[float, float] | None:
    """
    Fit a circle to 2D points using RANSAC and return the circle center (x, y).
    Returns None when fitting fails or produces an invalid model.

    :param points: Array of (x, y) points, shape (N, 2)
    :param n_iterations: Number of RANSAC iterations
    :param threshold: Inlier distance threshold (in same units as points)
    :return: Circle center (x, y) or None if fitting failed
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
    if model is not None:
        x, y, radius = model.params
        if radius > 0 and np.isfinite([x, y, radius]).all():
            return x, y

    return None


def _get_bounding_box_center(mask: NDArray[np.bool_]) -> tuple[float, float]:
    """Return center of bounding box for True values in mask."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return mask.shape[1] / 2, mask.shape[0] / 2
    return (cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2


def _set_map_center(
    data: ScanMap2DArray,
    use_circle: bool = False,
) -> tuple[float, float]:
    """Compute map center from data bounds or circle fit.

    :param data: Height map array
    :param use_circle: Use RANSAC circle fitting (for breech face impressions)
    :return: Center position (col, row) in pixel coordinates
    """
    valid_mask = ~np.isnan(data)

    if use_circle:
        edge_points = _get_mask_edge_points(valid_mask)
        center = _fit_circle_ransac(edge_points)
        if center is not None:
            return center

    # Fallback: bounding box center
    return _get_bounding_box_center(valid_mask)


def _estimate_plane_tilt_degrees(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
) -> tuple[float, float, NDArray[np.floating]]:
    """
    Estimate best-fit plane and return tilt angles in degrees + residuals.

    Fits z = ax + by + c using least squares.

    :param x: X coordinates
    :param y: Y coordinates
    :param z: Z values at each (x, y)
    :return: (tilt_x_deg, tilt_y_deg, residuals)
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)

    tilt_x_deg = np.degrees(np.arctan(a))
    tilt_y_deg = np.degrees(np.arctan(b))
    residuals = z - (a * x + b * y + c)

    return tilt_x_deg, tilt_y_deg, residuals


def _get_valid_coordinates(
    mark_image: MarkImage,
    center: tuple[float, float],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Get x, y, z coordinates of valid pixels in meters, centered at origin."""
    valid_mask = ~np.isnan(mark_image.data)
    rows, cols = np.where(valid_mask)

    x = cols * mark_image.scale_x - center[0]
    y = rows * mark_image.scale_y - center[1]
    z = mark_image.data[valid_mask]

    return x, y, z


def _adjust_for_plane_tilt_degrees(
    mark_image: MarkImage,
    center: tuple[float, float],
) -> MarkImage:
    """
    Remove plane tilt from mark image and adjust scale factors.

    :param mark_image: Mark image to level
    :param center: Center position in meters
    :return: Leveled mark image with adjusted scale
    """
    x, y, z = _get_valid_coordinates(mark_image, center)

    if len(x) < 3:
        raise ValueError("Need at least 3 valid points to estimate plane tilt")

    tilt_x_deg, tilt_y_deg, residuals = _estimate_plane_tilt_degrees(x, y, z)

    data = mark_image.data.copy()
    data[~np.isnan(mark_image.data)] = residuals

    scale_x_new = mark_image.scale_x / np.cos(np.radians(tilt_x_deg))
    scale_y_new = mark_image.scale_y / np.cos(np.radians(tilt_y_deg))
    center_new = (
        center
        * np.array([scale_x_new, scale_y_new])
        / np.array([mark_image.scale_x, mark_image.scale_y])
    )

    return mark_image.model_copy(
        update={
            "data": data,
            "scale_x": scale_x_new,
            "scale_y": scale_y_new,
            "center": center_new,
        }
    )


def _apply_anti_aliasing(
    mark_image: MarkImage,
    target_spacing: tuple[float, float],
) -> tuple[MarkImage, tuple[Optional[float], Optional[float]]]:
    """Apply anti-aliasing filter before downsampling.

    Anti-aliasing prevents high-frequency content from appearing as false
    low-frequency patterns (aliasing artifacts) when resampling to a coarser
    resolution. This is achieved by low-pass filtering to remove frequencies
    above the Nyquist limit of the target resolution before downsampling.

    The filter is applied when downsampling by more than 1.5x
    (target spacing > 1.5 * current spacing).

    :param mark_image: Input mark image
    :param target_spacing: Target pixel spacing in meters
    :return: Tuple of (filtered image, cutoff applied) or (original image, None)
    """
    downsample_ratio = (
        target_spacing[0] / mark_image.scale_x,
        target_spacing[1] / mark_image.scale_y,
    )

    # Only filter if downsampling by >1.5x (matching MATLAB's 2*1.5 = 3 threshold)
    if not any(r > 1.5 for r in downsample_ratio):
        return mark_image, (None, None)

    cutoffs = (
        downsample_ratio[0] * mark_image.scale_x,
        downsample_ratio[1] * mark_image.scale_y,
    )

    filtered_data = apply_gaussian_filter(
        mark_image.data,
        is_high_pass=False,
        cutoff_lengths=cutoffs,
    )

    return mark_image.model_copy(update={"data": filtered_data}), cutoffs


def _set_center(
    mark_image: MarkImage,
) -> tuple[float, float]:
    """Get global and local center coordinates from metadata or compute from data.

    :param mark_image: Input mark image
    :return: center_local (x, y) in meters
    """
    use_circle = mark_image.mark_type == MarkType.BREECH_FACE_IMPRESSION
    center_px = _set_map_center(mark_image.data, use_circle)
    center_local = (
        center_px[0] * mark_image.scale_x,
        center_px[1] * mark_image.scale_y,
    )
    return center_local


def preprocess_impression_mark(
    mark_image: MarkImage,
    params: PreprocessingImpressionParams,
) -> tuple[MarkImage, MarkImage]:
    """Preprocess trimmed impression image data.

    Processing steps:
    1. Set image center
    2. Crop to smallest size -> skip?
    3. Adjust pixel spacing based on sample tilt (optional)
    4. Level data
    5. Apply anti-aliasing filter
    6. Apply low-pass filter
    7. Resample to desired resolution
    8. Apply high-pass filter
    9. Re-level data

    :param mark_image: MarkImage for trimmed impression data
    :param params: Processing parameters
    :return: tuple[MarkImage, MarkImage] with filtered-and-leveled and just-leveled data
    """
    # 1. Set image center
    center_local = _set_center(mark_image)

    # 2. Crop mark
    cropped_image = crop_nan_borders(mark_image.data)
    mark_image = mark_image.model_copy(update={"data": cropped_image})

    # 3. Adjust pixel spacing based on sample tilt (optional)
    interpolated = False
    if params.adjust_pixel_spacing:
        mark_image = _adjust_for_plane_tilt_degrees(mark_image, center_local)

    # 4. Level data
    leveled_results = level_map(
        mark_image,
        terms=params.surface_terms,
        is_highpass=False,
        image_center=center_local,
    )
    fitted_surface = leveled_results.fitted_surface
    mark_image_leveled = mark_image.model_copy(
        update={"data": leveled_results.leveled_map}
    )

    # 5. Apply anti-aliasing filter
    mark_image_leveled_aa = mark_image_leveled.model_copy()
    cutoff_low = (None, None)
    if params.pixel_size is not None:
        mark_image_leveled_aa, cutoff_low = _apply_anti_aliasing(
            mark_image_leveled_aa, params.pixel_size
        )

    # 6. Apply low-pass filter (if no anti-aliasing is performed
    mark_image_filtered = mark_image_leveled_aa.model_copy()
    if params.lowpass_cutoff is not None:
        # Low-pass is configured - check if it's stronger than anti-aliasing
        if cutoff_low == (None, None) or params.lowpass_cutoff < min(
            c for c in cutoff_low if c is not None
        ):
            data_filtered = apply_gaussian_filter(
                mark_image_leveled.data,
                is_high_pass=False,
                cutoff_lengths=(params.lowpass_cutoff, params.lowpass_cutoff),
            )
            mark_image_filtered = mark_image_leveled.model_copy(
                update={"data": data_filtered}
            )

    # 7. Resample to desired resolution
    if params.pixel_size is not None:
        if not np.allclose(
            (mark_image_filtered.scale_x, mark_image_filtered.scale_y),
            params.pixel_size,
            rtol=1e-7,
        ):
            mark_image_filtered, _ = resample(
                mark_image_filtered, target_sampling_distance=params.pixel_size[0]
            )
            mark_image_leveled_aa, _ = resample(
                mark_image_leveled_aa, target_sampling_distance=params.pixel_size[0]
            )
            fitted_surface_image = mark_image_leveled.model_copy(
                update={"data": fitted_surface}
            )
            fitted_surface_image, _ = resample(
                fitted_surface_image, target_sampling_distance=params.pixel_size[0]
            )
            fitted_surface = fitted_surface_image.data
            interpolated = True

    # 8. Apply high-pass filter
    if params.highpass_cutoff is not None:
        data_filtered = apply_gaussian_filter(
            mark_image_filtered.data,
            is_high_pass=True,
            cutoff_lengths=(params.highpass_cutoff, params.highpass_cutoff),
        )
        mark_image_filtered = mark_image_filtered.model_copy(
            update={"data": data_filtered}
        )

    # 9. Re-level data
    leveled_results = level_map(
        mark_image_filtered, terms=params.surface_terms, is_highpass=False
    )
    mark_image_filtered = mark_image_filtered.model_copy(
        update={"data": leveled_results.leveled_map}
    )

    # For leveled-only data: add back the removed form, then level with rigid terms only
    mark_image_leveled_aa = mark_image_leveled_aa.model_copy(
        update={"data": mark_image_leveled_aa.data + fitted_surface}
    )
    rigid_terms = params.surface_terms & SurfaceTerms.PLANE
    leveled_results = level_map(
        mark_image_leveled_aa, terms=rigid_terms, is_highpass=False
    )
    mark_image_leveled_aa = mark_image_leveled_aa.model_copy(
        update={"data": leveled_results.leveled_map}
    )

    # Build output metadata
    mark_image.meta_data.update(asdict(params))
    mark_image.meta_data.update(
        {
            "center_g_x": 0,
            "center_g_y": 0,
            "center_l_x": center_local[0],
            "center_l_y": center_local[1],
            "is_crop": True,
            "is_prep": True,
            "is_interpolated": interpolated,
        }
    )
    return MarkImage(
        data=mark_image_filtered.data,
        scale_x=params.pixel_size[0],
        scale_y=params.pixel_size[1],
        mark_type=mark_image_filtered.mark_type,
        crop_type=mark_image_filtered.crop_type,
        meta_data=mark_image_filtered.meta_data
        | {"is_filtered": True, "is_leveled": True},
    ), MarkImage(
        data=mark_image_leveled_aa.data,
        scale_x=params.pixel_size[0],
        scale_y=params.pixel_size[1],
        mark_type=mark_image_leveled_aa.mark_type,
        crop_type=mark_image_leveled_aa.crop_type,
        meta_data=mark_image_leveled_aa.meta_data
        | {"is_filtered": False, "is_leveled": True},
    )
