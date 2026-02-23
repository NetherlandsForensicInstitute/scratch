"""
Per-cell registration: three-stage pipeline mirroring MATLAB's
cell_corr_angle  →  maps_register_corr  →  maps_register_fine (gradient / ECC).

Stage 1  (coarse angular sweep, angle-outer / cell-inner)
    For each candidate rotation angle, the entire comparison image is
    resampled once via bilinear interpolation.  Each reference cell is then
    matched against that resampled image using normalised cross-correlation
    (FFT-based match_template, equivalent to maps_register_trans_corr).
    This mirrors cell_corr_angle on full-resolution maps.

Stage 2  (narrow Fourier cross-correlation)
    Starting from the best angle found in Stage 1, a sub-pixel FFT
    cross-correlation (phase_cross_correlation) pins the translation to
    sub-pixel precision within a tight angular window, mirroring
    maps_register_corr.

Stage 3  (Lucas-Kanade / ECC gradient refinement)
    The Enhanced Correlation Coefficient (ECC) algorithm (Evangelidis &
    Psarakis 2008) refines [dx, dy, angle] iteratively using image-gradient
    updates, mirroring maps_register_fine with regAlgorithmFine='gradient'.
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
    find_grid_origin,
    generate_grid_centers,
)

# ECC convergence tolerances (passed to cv2.findTransformECC)
_ECC_MAX_ITER = 150
_ECC_TOL_SIM = 1e-4  # ECC value change threshold (maps to Par.convSim)


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

    The outer loop structure mirrors ``cell_corr_angle``: the comparison image
    is rotated once per candidate angle, then every reference cell is matched
    against that single rotated image (translation-only search at each angle).
    The best angle per cell is then passed to Stages 2 and 3.

    :param reference_map: Fixed surface map.
    :param comparison_map: Moving surface map.
    :param params: Algorithm parameters.  The angular sweep runs from
        ``search_angle_min`` to ``search_angle_max`` in steps of
        ``search_angle_step`` (all in degrees, centred on 0°).
    :returns: CellResult list for all cells that pass the fill-fraction check.
    """
    origin = find_grid_origin(reference_map, params)
    centers = generate_grid_centers(reference_map, origin, params)

    # Candidate rotation angles (degrees), centred on 0° as defined by params
    angles_deg = np.arange(
        params.search_angle_min,
        params.search_angle_max + params.search_angle_step,
        params.search_angle_step,
    )

    # Pre-extract valid reference patches (clip to image bounds, fill-check)
    patches = []  # (center, patch, fill_fraction)  or None per center
    spacing = reference_map.pixel_spacing
    cell_px = np.round(params.cell_size / spacing).astype(int)
    rows, cols = reference_map.height_map.shape

    for center in centers:
        py = int(round(center[1] / spacing[1] - cell_px[1] / 2))
        px = int(round(center[0] / spacing[0] - cell_px[0] / 2))
        y0, x0 = max(0, py), max(0, px)
        y1, x1 = min(rows, py + cell_px[1]), min(cols, px + cell_px[0])
        if y1 <= y0 or x1 <= x0:
            patches.append(None)
            continue
        patch = reference_map.height_map[y0:y1, x0:x1]
        # Fill fraction: clipped intersection area / full cell area.
        # MATLAB computes fill as the fraction of the full cell area that
        # overlaps the image (regardless of NaN holes within the image).
        clipped_area = (y1 - y0) * (x1 - x0)
        full_area = int(cell_px[0] * cell_px[1])
        fill = clipped_area / full_area
        if fill < params.minimum_fill_fraction:
            patches.append(None)
        else:
            patches.append((center, patch, fill))

    # Stage 1: angle-outer / cell-inner sweep
    # best_per_cell[i] = (score, angle_deg, comp_center_um)
    best_per_cell = [(-np.inf, 0.0, np.zeros(2)) for _ in centers]

    for angle_deg in angles_deg:
        # Rotate the entire comparison image once for this angle
        rotated = _rotate_image(comparison_map.height_map, angle_deg)
        clean_rotated = _fill_nan(rotated)

        for i, info in enumerate(patches):
            if info is None:
                continue
            center, patch, _ = info
            clean_patch = _fill_nan(patch) - np.nanmean(_fill_nan(patch))

            # match_template: full-image FFT cross-correlation
            # pad_input=False → output is smaller; index is top-left of best match
            if (
                clean_patch.shape[0] > clean_rotated.shape[0]
                or clean_patch.shape[1] > clean_rotated.shape[1]
            ):
                continue  # patch larger than rotated image (edge case)

            cc_map = match_template(clean_rotated, clean_patch, pad_input=False)
            iy, ix = np.unravel_index(np.argmax(cc_map), cc_map.shape)
            score = float(cc_map[iy, ix])

            if score > best_per_cell[i][0]:
                ph, pw = clean_patch.shape
                comp_center_um = np.array(
                    [
                        (ix + (pw - 1) / 2.0) * comparison_map.pixel_spacing[0],
                        (iy + (ph - 1) / 2.0) * comparison_map.pixel_spacing[1],
                    ]
                )
                best_per_cell[i] = (score, angle_deg, comp_center_um)

    # Stages 2 + 3: refine each valid cell
    results = []
    for i, info in enumerate(patches):
        if info is None:
            continue
        center, patch, fill = info
        coarse_score, coarse_angle_deg, coarse_comp_center = best_per_cell[i]

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


# ---------------------------------------------------------------------------
# Stage 2 + 3: per-cell refinement
# ---------------------------------------------------------------------------


def _refine_cell(
    center_ref_um: FloatArray1D,
    ref_patch: FloatArray2D,
    comp_map: SurfaceMap,
    coarse_angle_deg: float,
    coarse_comp_center_um: FloatArray1D,
    fill_fraction: float,
) -> CellResult:
    """
    Refine a single cell registration via FFT CC (Stage 2) then ECC (Stage 3).

    :param center_ref_um: Cell center on the reference map in µm, shape (2,).
    :param ref_patch: Clipped reference height patch (may be smaller than cell_size).
    :param comp_map: Moving surface map.
    :param coarse_angle_deg: Best angle from Stage 1 in degrees.
    :param coarse_comp_center_um: Best translation from Stage 1 in µm.
    :param fill_fraction: Reference cell fill fraction (pre-computed).
    :returns: CellResult.
    """
    spacing = comp_map.pixel_spacing

    # ---- Stage 2: sub-pixel FFT cross-correlation at the best angle ----
    comp_rotated = _rotate_image(comp_map.height_map, coarse_angle_deg)
    comp_clean = _fill_nan(comp_rotated)
    patch_clean = _fill_nan(ref_patch)
    ph, pw = patch_clean.shape

    if ph > comp_clean.shape[0] or pw > comp_clean.shape[1] or ph == 0 or pw == 0:
        # Can't refine; fall back to Stage 1 result
        return CellResult(
            center_reference=center_ref_um,
            center_comparison=coarse_comp_center_um,
            registration_angle=np.radians(coarse_angle_deg),
            area_cross_correlation_function_score=0.0,
            reference_fill_fraction=fill_fraction,
        )

    # Phase cross-correlation between the patch and the comparison crop
    # centred on the Stage 1 position estimate.
    cx_px = coarse_comp_center_um[0] / spacing[0]
    cy_px = coarse_comp_center_um[1] / spacing[1]
    # Extract comparison crop same size as patch, centred on coarse estimate
    y0c = int(round(cy_px - (ph - 1) / 2.0))
    x0c = int(round(cx_px - (pw - 1) / 2.0))
    comp_crop = _safe_crop(comp_clean, y0c, x0c, ph, pw)
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
    # shift = (row_shift, col_shift): comparison needs to move by -shift to align
    refined_cx = (x0c + (pw - 1) / 2.0 - shift[1]) * spacing[0]
    refined_cy = (y0c + (ph - 1) / 2.0 - shift[0]) * spacing[1]
    refined_angle_deg = coarse_angle_deg

    # ---- Stage 3: ECC / Lucas-Kanade gradient refinement ----
    ecc_center, ecc_angle_deg, ecc_score = _ecc_refine(
        ref_patch=patch_clean,
        comp_image=comp_clean,
        init_cx_px=refined_cx / spacing[0],
        init_cy_px=refined_cy / spacing[1],
        init_angle_deg=refined_angle_deg,
    )

    return CellResult(
        center_reference=center_ref_um,
        center_comparison=ecc_center * spacing,
        registration_angle=np.radians(ecc_angle_deg),
        area_cross_correlation_function_score=ecc_score,
        reference_fill_fraction=fill_fraction,
    )


# ---------------------------------------------------------------------------
# Stage 3: ECC (Enhanced Correlation Coefficient) — Lucas-Kanade variant
# ---------------------------------------------------------------------------


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

    NaN holes in both images are filled by nearest-neighbour interpolation
    before being passed to OpenCV.  The filled pixels are also supplied as
    an ``inputMask`` so that OpenCV excludes them from the gradient
    computation — this preserves the original behaviour of the hand-rolled
    ECC that operated only on ``np.isfinite`` pixels.

    The warp model is Euclidean (rotation + translation), identical to the
    previous implementation:
        [x', y'] = R(angle) * [x, y]^T + [tx, ty]^T

    All inputs and outputs are in pixel units; the caller is responsible
    for converting to physical coordinates (µm).

    :param ref_patch: Reference cell patch (may contain NaNs), shape (H, W).
    :param comp_image: Full comparison image (may contain NaNs), shape (M, N).
    :param init_cx_px: Initial comparison centre column (pixels).
    :param init_cy_px: Initial comparison centre row (pixels).
    :param init_angle_deg: Initial rotation angle (degrees).
    :returns: (center_pixels [cx, cy], angle_deg, accf_score).
              center_pixels is in *pixel* units; caller multiplies by spacing.
    """
    ph, pw = ref_patch.shape

    if ph < 3 or pw < 3:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # --- Crop comp_image to the same footprint as ref_patch ---
    # findTransformECC requires both images to be the same size.
    # We extract a window from comp_image centred on the Stage-2 estimate;
    # the ECC warp then refines the sub-pixel offset within that window.
    y0c = int(round(init_cy_px - (ph - 1) / 2.0))
    x0c = int(round(init_cx_px - (pw - 1) / 2.0))
    comp_crop_raw = _safe_crop(comp_image, y0c, x0c, ph, pw)
    if comp_crop_raw is None:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # --- Fill NaN holes via nearest-neighbour ---
    ref_filled, ref_nan_mask = _fill_nans_nearest(ref_patch)
    comp_filled, _ = _fill_nans_nearest(comp_crop_raw)

    # --- Normalise to [0, 1] float32 as required by findTransformECC ---
    ref_f32 = _norm01_f32(ref_filled)
    comp_f32 = _norm01_f32(comp_filled)

    # --- Build inputMask: pixels that were originally valid in the ref patch ---
    # (comp NaNs are not masked here — they are filled so OpenCV can sample
    # anywhere during warping; only ref pixels drive the gradient update)
    input_mask = (~ref_nan_mask).astype(np.uint8)
    if input_mask.sum() < 6:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    # --- Initial warp matrix ---
    # MOTION_EUCLIDEAN warp: [[cos, -sin, tx], [sin, cos, ty]]
    # Both images are the same size and comp_crop is already centred on the
    # Stage-2 estimate, so the initial displacement within the crop is zero.
    # The rotation is seeded from the Stage-1/2 angle.
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
        # Note: gaussFiltSize and inputMask must be passed positionally in OpenCV 4.x
        ecc_val, warp_out = cv2.findTransformECC(
            ref_f32,
            comp_f32,
            warp_init,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            input_mask,
            1,  # gaussFiltSize=1 (no smoothing — scan data is already smooth)
        )
    except cv2.error:
        # ECC did not converge — return the Stage 2 estimate with zero score
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    angle_out_deg = float(np.degrees(np.arctan2(warp_out[1, 0], warp_out[0, 0])))
    # tx/ty is the displacement of the comp crop centre relative to ref patch centre.
    # Add back the absolute position of the comp crop top-left to get pixel coords.
    cx_out = x0c + (pw - 1) / 2.0 + float(warp_out[0, 2])
    cy_out = y0c + (ph - 1) / 2.0 + float(warp_out[1, 2])

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
              *nan_mask* is a boolean array that is True where the original
              image was NaN.
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

    Returns None if the crop does not overlap the image at all, or if the
    image is smaller than the requested crop in both dimensions (no padding
    possible).  Otherwise pads out-of-bounds edges with zeros.
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
