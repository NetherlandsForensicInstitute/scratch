"""
Per-cell registration: three-stage pipeline.

Stage 1  (coarse angular sweep)
    The comparison image is rotated once per candidate angle.  For each
    rotation, every reference cell is matched against the rotated image using
    normalised cross-correlation (NCC).  NCC preserves frequency amplitudes,
    so its peak value is a true similarity score in [-1, 1].  This makes it
    possible to compare scores across different angles and reliably pick the
    best rotation for each cell.

Stage 2  (sub-pixel translation refinement)
    Starting from the best angle found in Stage 1, phase cross-correlation
    pins the translation to sub-pixel precision.  Unlike plain NCC, phase
    cross-correlation divides out the amplitude of every frequency component,
    keeping only the phase difference.  By the Fourier shift theorem, a
    translation in the spatial domain is a linear phase ramp in the frequency
    domain, so the result is a sharp spike whose location gives the shift with
    high precision.  The trade-off is that the peak height is no longer a
    meaningful similarity score — which is fine here because we are no longer
    comparing across angle hypotheses.

Stage 3  (ECC gradient refinement)
    The Enhanced Correlation Coefficient algorithm (Evangelidis & Psarakis
    2008) jointly refines translation and rotation by iteratively solving a
    linearised gradient system.  It is seeded from the Stage 2 translation and
    the Stage 1 angle, and converges to a sub-pixel, sub-degree estimate while
    also returning the final ECC score used for cell acceptance.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate

from container_models.base import FloatArray1D, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
)
from conversion.surface_comparison.grid import generate_grid_centers
from conversion.surface_comparison.utils import (
    meters_to_pixels,
    compute_top_left_pixel_of_cell,
)

# ECC convergence tolerances (passed to cv2.findTransformECC)
_ECC_MAX_ITER = 150
_ECC_TOL_SIM = 1e-4  # minimum ECC value change per iteration to continue


def register_cells(
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
) -> list[Cell]:
    """
    Run the three-stage per-cell registration pipeline.

    The comparison image is rotated once per candidate angle; every reference
    cell is then matched against that single rotated image.  This amortizes
    the cost of rotation across all cells.  The best angle per cell is passed
    to Stages 2 and 3 for sub-pixel refinement.

    :param reference_image: Fixed surface map.
    :param comparison_image: Moving surface map.
    :param params: Algorithm parameters.  The angular sweep runs from
        ``search_angle_min`` to ``search_angle_max`` in steps of
        ``search_angle_step`` (all in degrees, centred on 0°).
    :returns: Cell list for all cells that pass the fill-fraction check.
    """
    nrows, ncols = reference_image.data.shape
    origin = np.array(
        [
            np.ceil(ncols / 2) * reference_image.scale_x,
            np.ceil(nrows / 2) * reference_image.scale_y,
        ]
    )
    centers = generate_grid_centers(reference_image, origin, params)

    # Pre-extract reference cell patches and compute fill fractions.
    # Only cells that sufficiently overlap the image are kept.
    pixel_spacing = reference_image.pixel_spacing
    cell_size_px = meters_to_pixels(params.cell_size, pixel_spacing)
    n_rows, n_cols = reference_image.data.shape

    valid_cells = []

    for center in centers:
        row, col = compute_top_left_pixel_of_cell(center, cell_size_px, pixel_spacing)
        top_row, left_col = max(0, row), max(0, col)

        # lower/right corner of cell, in pixels
        bottom_row = min(n_rows, row + cell_size_px[1])
        right_col = min(n_cols, col + cell_size_px[0])

        if bottom_row <= top_row or right_col <= left_col:
            continue
        cell_data = reference_image.data[top_row:bottom_row, left_col:right_col]

        # Fill fraction is the fraction of the full cell area that overlaps
        # the image, regardless of NaN holes within the image.
        clipped_area = (bottom_row - top_row) * (right_col - left_col)
        full_area = int(cell_size_px[0] * cell_size_px[1])
        fill_fraction = clipped_area / full_area
        if fill_fraction < params.minimum_fill_fraction:
            continue

        reference_center = (
            (left_col + right_col - 1) / 2 * pixel_spacing[0],
            (bottom_row + top_row - 1) / 2 * pixel_spacing[1],
        )  # (x,y) in meters

        # TODO centers in pixels
        cell = Cell(
            center_reference=reference_center,
            cell_data=cell_data,
            fill_fraction_reference=fill_fraction,
        )
        # center in meters, 2D np.array, fill_fraction, initial score, initial angle, translation in meters
        valid_cells.append(cell)

    # ---- Stage 1: Sweep over angles_deg and store parameters for best cross_correlation per cell
    angles_deg = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    for angle_deg in angles_deg:
        rotated = _rotate_comparison_image(comparison_image.data, float(angle_deg))

        for cell in valid_cells:
            nan_filled_cell_data = _replace_nan_with_image_mean(cell.cell_data)

            cross_correlation_value, left_column, top_row = (
                _get_optimal_crosscorr_and_comparison_center(
                    rotated, nan_filled_cell_data
                )
            )

            if cell.best_score is None or cross_correlation_value > cell.best_score:
                _update_cell(
                    cell,
                    cross_correlation_value,
                    angle_deg,
                    left_column,
                    top_row,
                    comparison_image.pixel_spacing,
                )

    # ---- Stages 2 + 3: sub-pixel refinement per cell ----
    for cell in valid_cells:
        _fine_tune_cell(cell, comparison_image)

    return valid_cells


def _get_optimal_crosscorr_and_comparison_center(
    rotated: FloatArray2D, cell_data: FloatArray2D
) -> tuple[float, int, int]:
    mean_subtracted_rotated = _replace_nan_with_image_mean(rotated)
    mean_subtracted_cell_data = _replace_nan_with_image_mean(cell_data)
    """Compute optimal cross correlation and corresponding location of top-left idxs of mean_subtracted_cell_data.
        Mean_subtracted_rotated is reference.

    :param rotated: the rotated comparison image.
    :param cell_data: the cell data.
    :return: largest cross correlation value and corresponding translation in pixel space (y, x)
    """
    # match_template computes NCC via FFT; pad_input=False means the
    # output is (comp_height - cell_data_height + 1, comp_width - cell_data_width + 1)
    # and each index corresponds to the top-left corner of the cell_data position.
    cc_map = match_template(
        mean_subtracted_rotated, mean_subtracted_cell_data, pad_input=False
    )
    iy, ix = np.unravel_index(np.argmax(cc_map), cc_map.shape)
    score = float(cc_map[iy, ix])

    return score, ix, iy


def _update_cell(
    cell, cross_correlation_value, angle, left_column, top_row, pixel_spacing
):
    pixel_height, pixel_width = cell.cell_data.shape
    center_comparison_meters = np.array(
        [
            (left_column + (pixel_width - 1) / 2.0) * pixel_spacing[0],
            (top_row + (pixel_height - 1) / 2.0) * pixel_spacing[1],
        ]
    )
    cell.best_score = cross_correlation_value
    cell.angle_reference = angle
    cell.center_comparison = center_comparison_meters


def _fine_tune_cell(cell: Cell, comparison_image: ScanImage):
    """
    Refine a single cell registration via phase cross-correlation (Stage 2)
    then ECC (Stage 3).

    :param cell: a Cell instance with Stage 1 estimates populated (mutated in place).
    :param comparison_image: Full moving surface map.
    """
    pixel_spacing = comparison_image.pixel_spacing

    # Rotate the comparison image to the Stage 1 angle so that Stages 2 and 3
    # only need to refine the translation (and a small residual angle).
    comp_rotated = _rotate_comparison_image(comparison_image.data, cell.angle_reference)
    comp_mean_subtracted = _replace_nan_with_image_mean(comp_rotated)
    cell_data_mean_subtracted = _replace_nan_with_image_mean(cell.cell_data)
    pixel_height, pixel_width = cell_data_mean_subtracted.shape

    if (
        pixel_height > comp_mean_subtracted.shape[0]
        or pixel_width > comp_mean_subtracted.shape[1]
        or pixel_height == 0
        or pixel_width == 0
    ):
        cell.best_score = 0
        return

    # Crop the comparison image to the region around the Stage 1 estimate.
    # The crop is the same size as the reference cell_data so that phase
    # cross-correlation can be applied directly.
    x0c = int(
        round(cell.center_comparison[0] / pixel_spacing[0] - (pixel_width - 1) / 2.0)
    )
    y0c = int(
        round(cell.center_comparison[1] / pixel_spacing[1] - (pixel_height - 1) / 2.0)
    )
    comp_crop = _safe_crop(comp_mean_subtracted, y0c, x0c, pixel_height, pixel_width)

    if comp_crop is None:
        cell.best_score = 0
        return

    shift, _, _ = phase_cross_correlation(
        cell_data_mean_subtracted - cell_data_mean_subtracted.mean(),
        comp_crop - comp_crop.mean(),
        upsample_factor=10,
    )
    # shift = (row_shift, col_shift): the comp crop needs to move by -shift
    # to align with the reference cell_data.
    refined_cx = (x0c + (pixel_width - 1) / 2.0 - shift[1]) * pixel_spacing[0]
    refined_cy = (y0c + (pixel_height - 1) / 2.0 - shift[0]) * pixel_spacing[1]
    refined_angle_deg = cell.angle_reference

    # ---- Stage 3: ECC gradient refinement ----
    ecc_center, ecc_angle_deg, ecc_score = _ecc_refine(
        ref_cell_data=cell_data_mean_subtracted,
        comp_image=comp_mean_subtracted,
        init_cx_px=refined_cx / pixel_spacing[0],
        init_cy_px=refined_cy / pixel_spacing[1],
        init_angle_deg=refined_angle_deg,
    )

    cell.best_score = ecc_score
    cell.center_comparison = ecc_center * pixel_spacing
    cell.angle_reference = ecc_angle_deg


def _ecc_refine(
    ref_cell_data: FloatArray2D,
    comp_image: FloatArray2D,
    init_cx_px: float,
    init_cy_px: float,
    init_angle_deg: float,
) -> tuple[FloatArray1D, float, float]:
    """
    Refine cell registration using OpenCV's ECC algorithm
    (``cv2.findTransformECC`` with ``MOTION_EUCLIDEAN``).

    The warp model is Euclidean (rotation + translation):
        [x', y'] = R(angle) * [x, y]^T + [tx, ty]^T

    NaN holes are filled by nearest-neighbour interpolation before being
    passed to OpenCV. The filled pixels are excluded from the gradient
    computation via ``inputMask``, so only originally valid pixels contribute
    to the ECC update.

    All inputs and outputs are in pixel units; the caller is responsible
    for converting to physical coordinates (m).

    :param ref_cell_data: Reference cell data (may contain NaNs), shape (H, W).
    :param comp_image: Comparison image crop (may contain NaNs), shape (H, W).
    :param init_cx_px: Initial comparison center column in pixels (from Stage 2).
    :param init_cy_px: Initial comparison center row in pixels (from Stage 2).
    :param init_angle_deg: Initial rotation angle in degrees (from Stage 1).
    :returns: (center_pixels [cx, cy], angle_deg, ecc_score).
    """
    pixel_height, pixel_width = ref_cell_data.shape

    if pixel_height < 3 or pixel_width < 3:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # Crop comp_image to the same size as ref_cell_data, centred on the Stage 2
    # estimate.  findTransformECC requires both images to have the same shape,
    # and working with a small crop is faster than operating on the full image.
    y0c = int(round(init_cy_px - (pixel_height - 1) / 2.0))
    x0c = int(round(init_cx_px - (pixel_width - 1) / 2.0))
    comp_crop_raw = _safe_crop(comp_image, y0c, x0c, pixel_height, pixel_width)
    if comp_crop_raw is None:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    ref_filled, ref_nan_mask = _replace_nan_with_image_nearest_valid(ref_cell_data)
    comp_filled, _ = _replace_nan_with_image_nearest_valid(comp_crop_raw)

    # findTransformECC requires float32 input in [0, 1].
    ref_f32 = _norm01_f32(ref_filled)
    comp_f32 = _norm01_f32(comp_filled)

    # Exclude originally-NaN reference pixels from the gradient computation.
    input_mask = (~ref_nan_mask).astype(np.uint8)
    if input_mask.sum() < 6:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # The comp crop is already centred on the Stage 2 estimate, so the initial
    # displacement within the crop window is zero.  Only the rotation is seeded
    # from the Stage 1/2 result.
    cos_a = float(np.cos(np.radians(init_angle_deg)))
    sin_a = float(np.sin(np.radians(init_angle_deg)))
    warp_init = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        _ECC_MAX_ITER,
        _ECC_TOL_SIM,
    )

    try:
        # gaussFiltSize=1 disables pre-smoothing; scan data is already smooth.
        # inputMask and gaussFiltSize must be passed positionally in OpenCV 4.x.
        ecc_val, warp_out = cv2.findTransformECC(
            ref_f32,
            comp_f32,
            warp_init,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            input_mask,
            1,
        )
    except cv2.error:
        # ECC did not converge; return the Stage 2 estimate with zero score.
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    angle_out_deg = float(np.degrees(np.arctan2(warp_out[1, 0], warp_out[0, 0])))
    # warp_out[0, 2] and [1, 2] are the translation within the crop window.
    # Adding the crop origin recovers the absolute pixel position.
    cx_out = x0c + (pixel_width - 1) / 2.0 + float(warp_out[0, 2])
    cy_out = y0c + (pixel_height - 1) / 2.0 + float(warp_out[1, 2])

    return np.array([cx_out, cy_out]), angle_out_deg, float(ecc_val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_comparison_image(
    image: FloatArray2D, angle_deg_reference: float
) -> FloatArray2D:
    """Rotate image by angle_deg (degrees), NaN-padding, full-resolution."""
    if np.isclose(angle_deg_reference, 0.0):
        return image
    # rotate by negative angle since angle_deg_reference is defined as the rotation of the reference
    return rotate(image, -float(angle_deg_reference), preserve_range=True, cval=np.nan)


def _replace_nan_with_image_nearest_valid(
    image: FloatArray2D,
) -> tuple[FloatArray2D, np.ndarray]:
    """
    Replace NaN pixels with the value of the nearest valid pixel.

    Uses ``scipy.ndimage.distance_transform_edt`` to find the nearest
    non-NaN neighbour for every NaN pixel in O(N) time.

    :param image: Input array, may contain NaNs.
    :returns: ``(filled, nan_mask)`` where *filled* has no NaNs and
              *nan_mask* is True where the original image was NaN.
    """
    nan_mask = np.isnan(image)
    if not nan_mask.any():
        return image.astype(np.float64), nan_mask
    _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True)
    filled = image.astype(np.float64).copy()
    filled[nan_mask] = image[tuple(nearest_idx[:, nan_mask])]
    return filled, nan_mask


def _norm01_f32(image: FloatArray2D) -> np.ndarray:
    """
    Linearly scale *image* to [0, 1] and return as float32.

    ``cv2.findTransformECC`` requires single-channel float32 input.
    """
    mn, mx = float(image.min()), float(image.max())
    return ((image - mn) / (mx - mn + 1e-30)).astype(np.float32)


def _replace_nan_with_image_mean(image: FloatArray2D) -> FloatArray2D:
    """Replace NaN with the image mean (or 0 if all-NaN)."""
    if not np.any(np.isnan(image)):
        return image.astype(np.float64)
    mean = float(np.nanmean(image)) if not np.all(np.isnan(image)) else 0.0
    return np.where(np.isnan(image), mean, image).astype(np.float64)


def _safe_crop(
    image: FloatArray2D,
    top_row: int,
    left_col: int,
    height: int,
    width: int,
) -> FloatArray2D | None:
    """
    Extract a (height × width) crop from image starting at (top_row, left_col).

    Returns None if the crop does not overlap the image at all.
    Out-of-bounds edges are zero-padded.
    """
    img_h, img_w = image.shape
    if (
        top_row >= img_h
        or left_col >= img_w
        or top_row + height <= 0
        or left_col + width <= 0
    ):
        return None
    out = np.zeros((height, width), dtype=np.float64)
    r0s = max(0, -top_row)
    r0d = max(0, top_row)
    c0s = max(0, -left_col)
    c0d = max(0, left_col)
    r1s = min(height, img_h - top_row)
    r1d = min(img_h, top_row + height)
    c1s = min(width, img_w - left_col)
    c1d = min(img_w, left_col + width)
    out[r0s:r1s, c0s:c1s] = image[r0d:r1d, c0d:c1d]
    return out
