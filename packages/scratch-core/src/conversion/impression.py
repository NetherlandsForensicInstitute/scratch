# from dataclasses import asdict, dataclass
#
# import numpy as np
# from numpy.typing import NDArray
# from scipy.constants import micro
# from scipy.ndimage import binary_erosion
# from skimage.measure import CircleModel, ransac
#
# from conversion.data_formats import MarkImage, MarkType
# from utils.array_definitions import ScanMap2DArray, MaskArray
#
#
# def _get_mask_edge_points(mask: MaskArray) -> NDArray[np.floating]:
#     """Get inner edge points of a binary mask in pixel coordinates.
#
#     :param mask: Binary mask array
#     :return: Array of (col, row) edge points in pixel indices
#     """
#
#     eroded = binary_erosion(mask)
#     edge = mask & ~eroded
#
#     rows, cols = np.where(edge)
#     return np.column_stack([cols, rows]).astype(float)
#
#
# def _fit_circle_ransac(
#     points: NDArray[np.floating],
#     n_iterations: int = 1000,
#     threshold: float = 1.0,
# ) -> tuple[float, float] | None:
#     """Fit circle to points using RANSAC.
#
#     :param points: Array of (x, y) points, shape (N, 2)
#     :param n_iterations: Number of RANSAC iterations
#     :param threshold: Inlier distance threshold (in same units as points)
#     :return: Circle center (x, y) or None if fitting failed
#     """
#
#     try:
#         model, _ = ransac(
#             points,
#             CircleModel,
#             min_samples=3,
#             residual_threshold=threshold,
#             max_trials=n_iterations,
#         )
#         if model is not None:
#             return (model.params[0], model.params[1])
#     except Exception:
#         pass
#
#     return None
#
#
# def _set_map_center(
#     data: ScanMap2DArray,
#     use_circle: bool = False,
# ) -> tuple[float, float]:
#     """Compute map center from data bounds or circle fit.
#
#     :param data: Height map array
#     :param use_circle: Use RANSAC circle fitting (for breech face impressions)
#     :return: Center position (col, row) in pixel coordinates
#     """
#     valid_mask = ~np.isnan(data)
#
#     if use_circle:
#         edge_points = _get_mask_edge_points(valid_mask)
#         if len(edge_points) >= 3:
#             center = _fit_circle_ransac(edge_points)
#             if center is not None:
#                 return center
#
#     # Fallback: bounding box center
#     rows, cols = np.where(valid_mask)
#     if len(rows) == 0:
#         return data.shape[1] / 2, data.shape[0] / 2
#
#     return (cols.min() + cols.max() + 1) / 2, (rows.min() + rows.max() + 1) / 2
#
#
# @dataclass
# class PreprocessingImpressionParams:
#     """Processing parameters for NIST preprocessing.
#
#     :param pixel_size: Target pixel spacing in meters for resampling
#     :param adjust_pixel_spacing: Adjust pixel spacing based on sample tilt
#     :param level_offset: Remove constant offset
#     :param level_tilt: Remove linear tilt
#     :param level_2nd: Remove second-order terms
#     :param interp_method: Interpolation method ('nearest', 'linear', 'cubic')
#     :param highpass_cutoff: High-pass filter cutoff length in meters (None to disable)
#     :param lowpass_cutoff: Low-pass filter cutoff length in meters (None to disable)
#     """
#
#     pixel_size: tuple[float, float] = (
#         1.0,
#         1.0,
#     )  # Not set anywhere, always (1,1) or even (np.nan, np.nan)?
#     adjust_pixel_spacing: bool = True  # Not set anywhere, always False? set when initialising NIST params, always True?
#
#     # in Java Preprocessing/Impression parameter group
#     level_offset: bool = True
#     level_tilt: bool = True
#     level_2nd: bool = True
#     interp_method: str = "cubic"
#     highpass_cutoff: float | None = 250.0 * micro
#     lowpass_cutoff: float | None = 5.0 * micro
#     regression_order_high: int = 2
#     regression_order_low: int = 0
#     n_contiguous = None  # Not needed?
#     min_outlier_slope = None  # Not needed?
#     min_pixel_area = None  # Not needed?
#
#     @property
#     def surface_terms(self) -> SurfaceTerms:
#         """Convert leveling flags to SurfaceTerms."""
#         terms = SurfaceTerms.NONE
#         if self.level_offset:
#             terms |= SurfaceTerms.OFFSET
#         if self.level_tilt:
#             terms |= SurfaceTerms.TILT_X | SurfaceTerms.TILT_Y
#         if self.level_2nd:
#             terms |= SurfaceTerms.ASTIG_45 | SurfaceTerms.DEFOCUS | SurfaceTerms.ASTIG_0
#         return terms
#
#
# def _estimate_plane_tilt_degrees(x, y, z) -> tuple[float, float, float]:
#     """Estimate best-fit plane and return tilt angles in degrees + residuals.
#
#     :return: Tuple of (tilt_x_deg, tilt_y_deg)
#     """
#     # Fit plane: z = ax + by + c
#     A = np.column_stack([x, y, np.ones_like(x)])
#     coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
#
#     # Compute tilt angles in degrees
#     tilt_x_deg = np.degrees(np.arctan(coeffs[0]))
#     tilt_y_deg = np.degrees(np.arctan(coeffs[1]))
#     residuals = z - (coeffs[0] * x + coeffs[1] * y + coeffs[2])
#     return tilt_x_deg, tilt_y_deg, residuals
#
#
# def _adjust_for_plane_tilt_degrees(
#     mark_image: MarkImage,
#     center: tuple[float, float],
# ) -> MarkImage:
#     """Estimate best-fit plane and return tilt angles in degrees + residuals.
#
#     :param mark_image: Mark image
#     :param center: Center position in meters
#     :return: Tuple of ((tilt_x_deg, tilt_y_deg), leveled data)
#     """
#     valid_mask = ~np.isnan(mark_image.data)
#     rows, cols = np.where(valid_mask)
#     data = mark_image.data.copy()
#
#     if len(rows) < 3:
#         return mark_image
#
#     x = cols * mark_image.scale_x - center[0]
#     y = rows * mark_image.scale_y - center[1]
#     z = data[valid_mask]
#
#     tilt_x_deg, tilt_y_deg, residuals = _estimate_plane_tilt_degrees(x, y, z)
#
#     # Compute residuals
#     data[valid_mask] = residuals
#     adjusted_scale_x = mark_image.scale_x / np.cos(np.radians(tilt_x_deg))
#     adjusted_scale_y = mark_image.scale_y / np.cos(np.radians(tilt_y_deg))
#
#     return MarkImage(
#         data=data,
#         scale_x=adjusted_scale_x,
#         scale_y=adjusted_scale_y,
#         mark_type=mark_image.mark_type,
#         crop_type=mark_image.crop_type,
#         meta_data=mark_image.meta_data,
#     )
#
#
# def _apply_anti_aliasing(
#     data: MarkImage,
#     target_spacing: tuple[float, float],
# ) -> tuple[MarkImage, tuple[float, float]]:
#     """Apply anti-aliasing filter before downsampling.
#
#     Anti-aliasing prevents high-frequency content from appearing as false
#     low-frequency patterns (aliasing artifacts) when resampling to a coarser
#     resolution. This is achieved by low-pass filtering to remove frequencies
#     above the Nyquist limit of the target resolution before downsampling.
#
#     The filter is applied when:
#     - Downsampling by more than 1.5x (target spacing > 1.5 * current spacing)
#     - No existing low-pass filter with sufficient cutoff is already applied
#
#     :param data: Input array
#     :param target_spacing: Target pixel spacing in meters (assumes square pixels)
#     :return: Tuple of (filtered data, updated cutoff_low)
#     """
#     aa_ratio = (target_spacing[0] / data.scale_x, target_spacing[1] / data.scale_y)
#     new_cutoff = (aa_ratio[0] * data.scale_x, aa_ratio[1] * data.scale_y)
#
#     # Only filter if downsampling by >1.5x and existing filter insufficient (10% tolerance)
#     if not any(r > 1.5 for r in aa_ratio):
#         return data, (np.nan, np.nan)
#
#     return apply_gaussian_filter(
#         data, is_highpass=False, cutoff_length=new_cutoff
#     ), new_cutoff
#
#
# def preprocess_impression_mark(
#     mark_image: MarkImage,  # MarkImage
#     params: PreprocessingImpressionParams,
# ) -> tuple[MarkImage, MarkImage]:
#     """Preprocess trimmed impression image data.
#
#     Processing steps:
#     1. Set image center
#     2. Crop to smallest size
#     3. Adjust pixel spacing based on sample tilt (optional)
#     4. Level data
#     5. Apply low-pass / anti-aliasing filter
#     6. Resample to desired resolution
#     7. Apply high-pass filter
#     8. Re-level data
#
#     :param mark_image: MarkImage for trimmed impression data
#     :param params: Processing parameters
#     :return: tuple[MarkImage, MarkImage] with filtered-and-leveled and just-leveled data
#     """
#     # Extract existing center from metadata
#     center_global, center_local = _set_center(mark_image)
#
#     # Adjust pixel spacing for tilt
#     interpolated = False
#     if params.adjust_pixel_spacing:
#         mark_image = _adjust_for_plane_tilt_degrees(mark_image, center_local)
#
#     # Level data with full surface terms
#     mark_image, fitted_surface = _level_map_placeholder(
#         mark_image, params.surface_terms
#     )
#
#     # Apply anti-aliasing filter if downsampling
#     data_leveled = mark_image.model_copy()
#     cutoff_low = (np.nan, np.nan)
#     if params.pixel_size is not None:
#         data_leveled, cutoff_low = _apply_anti_aliasing(mark_image, params.pixel_size)
#
#     # Apply low-pass filter to original, or use anti-aliased if sufficient
#     data_filtered = data_leveled.model_copy()
#     if params.lowpass_cutoff is not None and (
#         any(np.isnan(cutoff_low)) or params.lowpass_cutoff >= max(cutoff_low)
#     ):
#         data_filtered = apply_gaussian_filter(
#             mark_image,
#             is_high_pass=False,
#             cutoff_length=(params.lowpass_cutoff, params.lowpass_cutoff),
#         )
#
#     # Resample to target resolution
#     if params.pixel_size is not None:
#         if not np.allclose(
#             (data_filtered.scale_x, data_filtered.scale_y), params.pixel_size, rtol=1e-7
#         ):
#             data_filtered = _resample_array_placeholder(
#                 data_filtered, params.pixel_size, params.interp_method
#             )
#             data_leveled = _resample_array_placeholder(
#                 data_leveled, params.pixel_size, params.interp_method
#             )
#             pixel_spacing = params.pixel_size
#             interpolated = True
#
#     # Apply high-pass filter
#     if params.highpass_cutoff is not None:
#         data_filtered = apply_gaussian_filter(
#             data_filtered,
#             is_high_pass=True,
#             cutoff_length=(params.highpass_cutoff, params.highpass_cutoff),
#         )
#
#     # Re-level filtered data with full terms
#     data_filtered, _ = _level_map_placeholder(data_filtered, params.surface_terms)
#
#     # For leveled-only data: add back the removed form, then level with rigid terms only
#     data_leveled = data_leveled.model_copy(
#         update={"data": data_leveled.data + fitted_surface}
#     )
#     rigid_terms = params.surface_terms & SurfaceTerms.PLANE
#     data_leveled, _ = _level_map_placeholder(data_leveled, rigid_terms)
#
#     # Build output metadata
#     mark_image.meta_data.update(asdict(params))
#     mark_image.meta_data.update(
#         {
#             "center_g_x": center_global[0],
#             "center_g_y": center_global[1],
#             "center_l_x": center_local[0],
#             "center_l_y": center_local[1],
#             "is_crop": True,
#             "is_prep": True,
#             "is_interpolated": interpolated,
#         }
#     )
#     return MarkImage(
#         data=data_filtered.data,
#         scale_x=params.pixel_size[0],
#         scale_y=params.pixel_size[1],
#         mark_type=mark_image.mark_type,
#         crop_type=mark_image.crop_type,
#         meta_data=mark_image.meta_data | {"is_filtered": True, "is_leveled": True},
#     ), MarkImage(
#         data=data_leveled.data,
#         scale_x=params.pixel_size[0],
#         scale_y=params.pixel_size[1],
#         mark_type=mark_image.mark_type,
#         crop_type=mark_image.crop_type,
#         meta_data=mark_image.meta_data | {"is_filtered": False, "is_leveled": True},
#     )
#
#
# def _set_center(
#     mark_image: MarkImage,
# ) -> tuple[tuple[float, float], tuple[float, float]]:
#     center_local = (
#         mark_image.meta_data.get("center_l_x", 0.0),
#         mark_image.meta_data.get("center_l_y", 0.0),
#     )
#     center_global = (
#         mark_image.meta_data.get("center_g_x", 0.0),
#         mark_image.meta_data.get("center_g_y", 0.0),
#     )
#     # Check if breech face impression
#     use_circle = (
#         hasattr(mark_image, "mark_type")
#         and mark_image.mark_type == MarkType.BREECH_FACE_IMPRESSION
#     )
#
#     # Set center if not specified
#     if center_local == (0.0, 0.0):
#         center_px = _set_map_center(mark_image.data, use_circle)
#         center_local = (
#             center_px[0] * mark_image.scale_x,
#             center_px[1] * mark_image.scale_y,
#         )
#         center_global = (0.0, 0.0)
#     return center_global, center_local
