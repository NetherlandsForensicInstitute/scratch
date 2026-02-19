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

import numpy as np
from scipy.ndimage import map_coordinates
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

# ECC convergence tolerances (matching MATLAB maps_register_fine defaults)
_ECC_MAX_ITER = 150
_ECC_TOL_POS = 0.1  # pixels – maps to Par.convPosPix
_ECC_TOL_ANGLE = 1e-3  # radians – maps to Par.convAngle
_ECC_TOL_SIM = 1e-4  # ACCF change – maps to Par.convSim


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
    cell_px = (params.cell_size / spacing).astype(int)
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
        fill = np.count_nonzero(~np.isnan(patch)) / (cell_px[0] * cell_px[1])
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
            params=params,
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
        spacing=spacing,
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
    spacing: FloatArray1D,
) -> tuple[FloatArray1D, float, float]:
    """
    Refine cell registration using the ECC gradient algorithm.

    Direct Python translation of the 'gradient' branch inside
    maps_register_fine (Evangelidis & Psarakis 2008, equation 19).

    The warp model is affine-rotation + translation:
      [x', y'] = R(angle) * [x - cx, y - cy]^T + [cx, cy]^T + [dx, dy]^T

    Parameters are updated by solving the linearised ECC system at each
    iteration until convergence or the iteration limit is reached.

    :param ref_patch: Reference cell patch (NaN-filled to zero), shape (H, W).
    :param comp_image: Full comparison image (NaN-filled), shape (M, N).
    :param init_cx_px: Initial comparison center column (pixels).
    :param init_cy_px: Initial comparison center row (pixels).
    :param init_angle_deg: Initial rotation angle (degrees).
    :param spacing: Pixel spacing [dx, dy] in µm (used only for output conversion).
    :returns: (center_pixels [cx, cy], angle_deg, accf_score).
              center_pixels is in *pixel* units; caller multiplies by spacing.
    """
    ph, pw = ref_patch.shape
    # Reference pixel grid (row, col) relative to patch center
    pr = np.arange(ph, dtype=np.float64) - (ph - 1) / 2.0
    pc = np.arange(pw, dtype=np.float64) - (pw - 1) / 2.0
    grid_c, grid_r = np.meshgrid(pc, pr)  # both (ph, pw)

    # Flatten and mask to valid pixels only
    ref_flat = ref_patch.ravel()
    valid = np.isfinite(ref_flat)
    if valid.sum() < 6:
        return np.array([init_cx_px, init_cy_px]), init_angle_deg, 0.0

    r_v = grid_r.ravel()[valid]
    c_v = grid_c.ravel()[valid]
    t1 = ref_flat[valid]
    t1 = t1 - t1.mean()

    cx = float(init_cx_px)
    cy = float(init_cy_px)
    angle_deg = float(init_angle_deg)

    comp_rows, comp_cols = comp_image.shape
    prev_accf = -np.inf

    for _ in range(_ECC_MAX_ITER):
        cos_a = np.cos(np.radians(angle_deg))
        sin_a = np.sin(np.radians(angle_deg))

        # Sample comparison at warped positions
        samp_c = cx + cos_a * c_v - sin_a * r_v
        samp_r = cy + sin_a * c_v + cos_a * r_v

        t2 = map_coordinates(
            comp_image,
            [samp_r, samp_c],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        t2 = t2 - t2.mean()

        denom = np.sqrt(np.dot(t1, t1) * np.dot(t2, t2))
        if denom < 1e-30:
            break
        accf = float(np.dot(t1, t2) / denom)

        if abs(accf - prev_accf) < _ECC_TOL_SIM:
            break
        prev_accf = accf

        # Gradient of t2 at sampled locations (central differences via
        # map_coordinates so we stay sub-pixel)
        eps = 0.5
        g_c = (
            map_coordinates(
                comp_image,
                [samp_r, samp_c + eps],
                order=1,
                mode="nearest",
                prefilter=False,
            )
            - map_coordinates(
                comp_image,
                [samp_r, samp_c - eps],
                order=1,
                mode="nearest",
                prefilter=False,
            )
        ) / (2 * eps)
        g_r = (
            map_coordinates(
                comp_image,
                [samp_r + eps, samp_c],
                order=1,
                mode="nearest",
                prefilter=False,
            )
            - map_coordinates(
                comp_image,
                [samp_r - eps, samp_c],
                order=1,
                mode="nearest",
                prefilter=False,
            )
        ) / (2 * eps)

        # Steepest-descent images for [dx(col), dy(row), d_angle]
        # Jacobian of the warp w.r.t. [dx_col, dy_row, d_angle_rad]
        #   d(samp_c)/d(dx_col)   = 1,   d(samp_r)/d(dx_col) = 0
        #   d(samp_c)/d(dy_row)   = 0,   d(samp_r)/d(dy_row) = 1
        #   d(samp_c)/d(d_angle)  = -sin_a*c_v - cos_a*r_v
        #   d(samp_r)/d(d_angle)  =  cos_a*c_v - sin_a*r_v
        sd_dx = g_c * 1.0 + g_r * 0.0
        sd_dy = g_c * 0.0 + g_r * 1.0
        sd_angle = g_c * (-sin_a * c_v - cos_a * r_v) + g_r * (
            cos_a * c_v - sin_a * r_v
        )

        # Build G = [sd_dx | sd_dy | sd_angle] (n_valid × 3)
        G = np.column_stack([sd_dx, sd_dy, sd_angle])
        # Zero-mean each column (projection operator normalisation)
        G = G - G.mean(axis=0)

        # ECC update (Evangelidis & Psarakis 2008, eq. 17-19).
        # Direct translation of the MATLAB gradient branch in maps_register_fine:
        #   vp1 = mG \ mm(:,1)
        #   vp2 = mG \ mm(:,2)
        #   ... compute lambda ...
        #   vdPar = mG \ (lambda*mm(:,1) - mm(:,2))
        # where mG is (n×3) and \ means least-squares.
        vp1, _, _, _ = np.linalg.lstsq(G, t1, rcond=None)  # (3,)
        vp2, _, _, _ = np.linalg.lstsq(G, t2, rcond=None)  # (3,)

        # Projection terms: t'*mG*vp = t' * Pg * t2  where Pg projects onto col(G)
        t1_t2 = float(np.dot(t1, t2))
        t1_Pg_t2 = float(t1 @ (G @ vp2))  # t1' * mG * vp2
        t1_Pg_t1 = float(t1 @ (G @ vp1))  # t1' * mG * vp1
        t2_t2 = float(np.dot(t2, t2))
        t2_Pg_t2 = float(t2 @ (G @ vp2))  # t2' * mG * vp2

        if t1_t2 > t1_Pg_t2:
            lam = (t2_t2 - t2_Pg_t2) / max(t1_t2 - t1_Pg_t2, 1e-30)
        else:
            lam_1 = np.sqrt(t2_Pg_t2 / max(t1_Pg_t1, 1e-30))
            lam_2 = (t1_Pg_t2 - t1_t2) / max(t1_Pg_t1, 1e-30)
            lam = max(float(lam_1), float(lam_2))

        # vdPar = mG \ (lambda*t1 - t2)   — G is (n×3), rhs is (n,)
        dp, _, _, _ = np.linalg.lstsq(G, lam * t1 - t2, rcond=None)  # (3,)

        d_cx, d_cy, d_angle_rad = dp
        cx += float(d_cx)
        cy += float(d_cy)
        angle_deg += float(np.degrees(d_angle_rad))

        # Convergence check
        if (
            abs(d_cx) < _ECC_TOL_POS
            and abs(d_cy) < _ECC_TOL_POS
            and abs(d_angle_rad) < _ECC_TOL_ANGLE
        ):
            break

    return np.array([cx, cy]), float(angle_deg), float(prev_accf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_image(image: FloatArray2D, angle_deg: float) -> FloatArray2D:
    """Rotate image by angle_deg (degrees), NaN-padding, full-resolution."""
    if np.isclose(angle_deg, 0.0):
        return image
    return rotate(image, -float(angle_deg), preserve_range=True, cval=np.nan)


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
