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
from conversion.surface_comparison.models import (
    SurfaceMap,
    CellResult,
    ComparisonParams,
)
from conversion.surface_comparison.grid import (
    _find_grid_origin,
    generate_grid_centers,
)
from conversion.surface_comparison.utils import (
    m_to_pixels,
    center_m_to_top_left_pixel,
)

# ECC convergence tolerances (passed to cv2.findTransformECC)
_ECC_MAX_ITER = 150
_ECC_TOL_SIM = 1e-4  # minimum ECC value change per iteration to continue


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def register_cells(
    reference_map: SurfaceMap,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
) -> list[CellResult]:
    """
    Run the three-stage per-cell registration pipeline.

    The comparison image is rotated once per candidate angle; every reference
    cell is then matched against that single rotated image.  This amortizes
    the cost of rotation across all cells.  The best angle per cell is passed
    to Stages 2 and 3 for sub-pixel refinement.

    :param reference_map: Fixed surface map.
    :param comparison_map: Moving surface map.
    :param params: Algorithm parameters.  The angular sweep runs from
        ``search_angle_min`` to ``search_angle_max`` in steps of
        ``search_angle_step`` (all in degrees, centred on 0°).
    :returns: CellResult list for all cells that pass the fill-fraction check.
    """
    origin = _find_grid_origin(reference_map, params)
    centers = generate_grid_centers(reference_map, origin, params)

    angles_deg = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    # Pre-extract reference patches and compute fill fractions.
    # Only cells that sufficiently overlap the image are kept.
    pixel_spacing = reference_map.pixel_spacing
    cell_size_px = m_to_pixels(params.cell_size, pixel_spacing)
    rows, cols = reference_map.data.shape

    valid_cells = []  # [center_um, patch, fill_fraction, best_score, best_angle_deg, best_comp_center_um]
    for center in centers:
        row, col = center_m_to_top_left_pixel(center, cell_size_px, pixel_spacing)
        row0, col0 = max(0, row), max(0, col)

        # lower/right corner of cell, in pixels
        row1 = min(rows, row + cell_size_px[1])
        col1 = min(cols, col + cell_size_px[0])
        if row1 <= row0 or col1 <= col0:
            continue
        patch = reference_map.data[row0:row1, col0:col1]

        # Fill fraction is the fraction of the full cell area that overlaps
        # the image, regardless of NaN holes within the image.
        clipped_area = (row1 - row0) * (col1 - col0)
        full_area = int(cell_size_px[0] * cell_size_px[1])
        fill_fraction = clipped_area / full_area
        if fill_fraction < params.minimum_fill_fraction:
            continue

        valid_cells.append([center, patch, fill_fraction, -np.inf, 0.0, np.zeros(2)])

    # ---- Stage 1: Sweep over angles_deg and store parameters for best cross_correlation per cell
    for angle_deg in angles_deg:
        rotated = _rotate_image(comparison_map.data, float(angle_deg))
        clean_rotated = _fill_nan(rotated)

        for cell in valid_cells:
            center, patch = cell[0], cell[1]
            nan_filled_patch = _fill_nan(patch)
            clean_patch = nan_filled_patch - np.nanmean(nan_filled_patch)

            if (
                clean_patch.shape[0] > clean_rotated.shape[0]
                or clean_patch.shape[1] > clean_rotated.shape[1]
            ):
                continue  # patch larger than rotated image (edge case)

            # match_template computes NCC via FFT; pad_input=False means the
            # output is (comp_height - patch_height + 1, comp_width - patch_width + 1)
            # and each index corresponds to the top-left corner of the patch position.
            cc_map = match_template(clean_rotated, clean_patch, pad_input=False)
            iy, ix = np.unravel_index(np.argmax(cc_map), cc_map.shape)
            score = float(cc_map[iy, ix])

            if score > cell[3]:
                pixel_height, pixel_width = clean_patch.shape
                cell[3] = score
                cell[4] = angle_deg
                cell[5] = np.array(
                    [
                        (ix + (pixel_width - 1) / 2.0)
                        * comparison_map.pixel_spacing[0],
                        (iy + (pixel_height - 1) / 2.0)
                        * comparison_map.pixel_spacing[1],
                    ]
                )

    # ---- Stages 2 + 3: sub-pixel refinement per cell ----
    results = []
    for (
        center,
        patch,
        fill,
        coarse_score,
        coarse_angle_deg,
        coarse_comp_center,
    ) in valid_cells:
        cell = _refine_cell(
            center_ref_um=center,
            ref_patch=patch,
            comp_map=comparison_map,
            coarse_angle_deg=coarse_angle_deg,
            coarse_comp_center_um=coarse_comp_center,
            fill_fraction=fill,
        )
        results.append(cell)

    return results


def _refine_cell(
    center_ref_um: FloatArray1D,
    ref_patch: FloatArray2D,
    comp_map: SurfaceMap,
    coarse_angle_deg: float,
    coarse_comp_center_um: FloatArray1D,
    fill_fraction: float,
) -> CellResult:
    """
    Refine a single cell registration via phase cross-correlation (Stage 2)
    then ECC (Stage 3).

    :param center_ref_um: Cell center on the reference map in m, shape (2,).
    :param ref_patch: Clipped reference height patch (may be smaller than cell_size).
    :param comp_map: Moving surface map.
    :param coarse_angle_deg: Best angle from Stage 1 in degrees.
    :param coarse_comp_center_um: Best translation from Stage 1 in m.
    :param fill_fraction: Reference cell fill fraction (pre-computed).
    :returns: CellResult.
    """
    pixel_spacing = comp_map.pixel_spacing

    # Rotate the comparison image to the Stage 1 angle so that Stages 2 and 3
    # only need to refine the translation (and a small residual angle).
    comp_rotated = _rotate_image(comp_map.data, coarse_angle_deg)
    comp_clean = _fill_nan(comp_rotated)
    patch_clean = _fill_nan(ref_patch)
    pixel_height, pixel_width = patch_clean.shape

    if (
        pixel_height > comp_clean.shape[0]
        or pixel_width > comp_clean.shape[1]
        or pixel_height == 0
        or pixel_width == 0
    ):
        return CellResult(
            center_reference=center_ref_um,
            center_comparison=coarse_comp_center_um,
            registration_angle=np.radians(coarse_angle_deg),
            area_cross_correlation_function_score=0.0,
            reference_fill_fraction=fill_fraction,
        )

    # Crop the comparison image to the region around the Stage 1 estimate.
    # The crop is the same size as the reference patch so that phase
    # cross-correlation can be applied directly.
    x0c = int(
        round(coarse_comp_center_um[0] / pixel_spacing[0] - (pixel_width - 1) / 2.0)
    )
    y0c = int(
        round(coarse_comp_center_um[1] / pixel_spacing[1] - (pixel_height - 1) / 2.0)
    )
    comp_crop = _safe_crop(comp_clean, y0c, x0c, pixel_height, pixel_width)

    if comp_crop is None:
        return CellResult(
            center_reference=center_ref_um,
            center_comparison=coarse_comp_center_um,
            registration_angle=np.radians(coarse_angle_deg),
            area_cross_correlation_function_score=0.0,
            reference_fill_fraction=fill_fraction,
        )

    shift, _, _ = phase_cross_correlation(
        patch_clean - patch_clean.mean(),
        comp_crop - comp_crop.mean(),
        upsample_factor=10,
    )
    # shift = (row_shift, col_shift): the comp crop needs to move by -shift
    # to align with the reference patch.
    refined_cx = (x0c + (pixel_width - 1) / 2.0 - shift[1]) * pixel_spacing[0]
    refined_cy = (y0c + (pixel_height - 1) / 2.0 - shift[0]) * pixel_spacing[1]
    refined_angle_deg = coarse_angle_deg

    # ---- Stage 3: ECC gradient refinement ----
    ecc_center, ecc_angle_deg, ecc_score = _ecc_refine(
        ref_patch=patch_clean,
        comp_image=comp_clean,
        init_cx_px=refined_cx / pixel_spacing[0],
        init_cy_px=refined_cy / pixel_spacing[1],
        init_angle_deg=refined_angle_deg,
    )

    return CellResult(
        center_reference=center_ref_um,
        center_comparison=ecc_center * pixel_spacing,
        registration_angle=np.radians(ecc_angle_deg),
        area_cross_correlation_function_score=ecc_score,
        reference_fill_fraction=fill_fraction,
    )


def _ecc_refine(
    ref_patch: FloatArray2D,
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

    :param ref_patch: Reference cell patch (may contain NaNs), shape (H, W).
    :param comp_image: Full comparison image (may contain NaNs), shape (M, N).
    :param init_cx_px: Initial comparison center column in pixels (from Stage 2).
    :param init_cy_px: Initial comparison center row in pixels (from Stage 2).
    :param init_angle_deg: Initial rotation angle in degrees (from Stage 1).
    :returns: (center_pixels [cx, cy], angle_deg, accf_score).
    """
    pixel_height, pixel_width = ref_patch.shape

    if pixel_height < 3 or pixel_width < 3:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # Crop comp_image to the same size as ref_patch, centred on the Stage 2
    # estimate.  findTransformECC requires both images to have the same shape,
    # and working with a small crop is faster than operating on the full image.
    y0c = int(round(init_cy_px - (pixel_height - 1) / 2.0))
    x0c = int(round(init_cx_px - (pixel_width - 1) / 2.0))
    comp_crop_raw = _safe_crop(comp_image, y0c, x0c, pixel_height, pixel_width)
    if comp_crop_raw is None:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    ref_filled, ref_nan_mask = _fill_nans_nearest(ref_patch)
    comp_filled, _ = _fill_nans_nearest(comp_crop_raw)

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


def _rotate_image(image: FloatArray2D, angle_deg: float) -> FloatArray2D:
    """Rotate image by angle_deg (degrees), NaN-padding, full-resolution."""
    if np.isclose(angle_deg, 0.0):
        return image
    return rotate(image, -float(angle_deg), preserve_range=True, cval=np.nan)


def _fill_nans_nearest(image: FloatArray2D) -> tuple[FloatArray2D, np.ndarray]:
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


def _fill_nan(image: FloatArray2D) -> FloatArray2D:
    """Replace NaN with the image mean (or 0 if all-NaN)."""
    if not np.any(np.isnan(image)):
        return image.astype(np.float64)
    mean = float(np.nanmean(image)) if not np.all(np.isnan(image)) else 0.0
    return np.where(np.isnan(image), mean, image).astype(np.float64)


def _safe_crop(
    image: FloatArray2D,
    row0: int,
    col0: int,
    height: int,
    width: int,
) -> FloatArray2D | None:
    """
    Extract a (height × width) crop from image starting at (row0, col0).

    Returns None if the crop does not overlap the image at all.
    Out-of-bounds edges are zero-padded.
    """
    img_h, img_w = image.shape
    if row0 >= img_h or col0 >= img_w or row0 + height <= 0 or col0 + width <= 0:
        return None
    out = np.zeros((height, width), dtype=np.float64)
    r0s = max(0, -row0)
    r0d = max(0, row0)
    c0s = max(0, -col0)
    c0d = max(0, col0)
    r1s = min(height, img_h - row0)
    r1d = min(img_h, row0 + height)
    c1s = min(width, img_w - col0)
    c1d = min(img_w, col0 + width)
    out[r0s:r1s, c0s:c1s] = image[r0d:r1d, c0d:c1d]
    return out
