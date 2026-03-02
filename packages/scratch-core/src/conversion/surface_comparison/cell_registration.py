"""
Cell registration for the NIST Congruent Matching Cells pipeline.

Clean implementation using library functions for each stage:

1. **Coarse angular sweep** — ``scipy.ndimage.rotate`` + ``skimage.feature.match_template``
   finds the best angle and approximate translation per cell.
2. **Sub-pixel translation** — ``skimage.registration.phase_cross_correlation``
   refines translation to sub-pixel accuracy at the best angle.
3. **ECC gradient refinement** — ``cv2.findTransformECC`` jointly optimises
   ``[dx, dy, θ]`` using the Enhanced Correlation Coefficient algorithm.
4. **Final ACCF** — correlation on processed/filtered data at the registered
   position, matching MATLAB's ``commonEval='final'``.

Coordinate convention
---------------------
Internally everything is in **pixel** coordinates ``[row, col]``.
Conversion to/from the application's ``[x, y]`` metre coordinates happens
only at the adapter boundary.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from scipy.ndimage import rotate
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation

# Minimum rotation angle (degrees) worth applying.
# Below this, rotation is skipped to avoid NaN spread from cubic interpolation
# at NaN boundaries.  0.01° corresponds to < 0.02 px shift at a 100 px radius.
_MIN_ROTATION_DEG = 0.01


def _rotate_nan_safe(
    data: np.ndarray,
    angle_deg: float,
    order: int = 3,
) -> np.ndarray:
    """
    Rotate ``data`` by ``angle_deg`` without spreading NaN.

    Strategy: replace NaN with 0, rotate the data and a binary validity
    mask separately, then re-apply NaN where the rotated mask falls
    below 0.5 (i.e. the pixel has insufficient valid-data support).
    """
    mask = ~np.isnan(data)
    filled = np.where(mask, data, 0.0)

    rotated_data = rotate(
        filled, angle_deg, reshape=False, order=order, mode="constant", cval=0.0
    )
    rotated_mask = rotate(
        mask.astype(np.float64),
        angle_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    )

    rotated_data[rotated_mask < 0.5] = np.nan
    return rotated_data


# =====================================================================
# Data structures
# =====================================================================


@dataclass
class CellBox:
    """A single cell's pixel bounding box and metadata."""

    row_start: int
    col_start: int
    row_end: int  # exclusive
    col_end: int  # exclusive
    center_row: float  # centre in pixels (0-based)
    center_col: float
    fill_fraction: float


@dataclass
class CellRegistrationResult:
    """Per-cell registration output."""

    center_reference: np.ndarray  # [row, col] in metres
    center_comparison: np.ndarray  # [row, col] in metres
    registration_angle: float  # radians
    accf: float
    fill_fraction: float
    is_valid: bool = True


# =====================================================================
# Stage 0: Cell grid generation
# =====================================================================


def _generate_cells(
    ref_data: np.ndarray,
    cell_size_px: tuple[int, int],
    min_fill: float,
) -> list[CellBox]:
    """
    Tile the reference map into cells and filter by valid-pixel fill fraction.

    Edge cells that are smaller than the nominal size are kept as long as the
    fill fraction (valid pixels / actual cell pixels) meets ``min_fill``.

    :param ref_data: Reference height map (may contain NaN).
    :param cell_size_px: Cell size ``(rows, cols)`` in pixels.
    :param min_fill: Minimum fraction of valid pixels to keep a cell.
    :returns: List of valid cells.
    """
    nrows, ncols = ref_data.shape
    cell_h, cell_w = cell_size_px
    valid_mask = ~np.isnan(ref_data)

    # Use integral image for fast fill-fraction computation
    integral = np.cumsum(np.cumsum(valid_mask.astype(np.float64), axis=0), axis=1)

    cells = []
    row_starts = list(range(0, nrows, cell_h))
    col_starts = list(range(0, ncols, cell_w))

    for r0 in row_starts:
        r1 = min(r0 + cell_h, nrows)
        for c0 in col_starts:
            c1 = min(c0 + cell_w, ncols)

            # Count valid pixels using integral image
            n_valid = integral[r1 - 1, c1 - 1]
            if r0 > 0:
                n_valid -= integral[r0 - 1, c1 - 1]
            if c0 > 0:
                n_valid -= integral[r1 - 1, c0 - 1]
            if r0 > 0 and c0 > 0:
                n_valid += integral[r0 - 1, c0 - 1]

            # Fill relative to actual cell area, not nominal
            actual_area = (r1 - r0) * (c1 - c0)
            fill = n_valid / actual_area
            if fill >= min_fill:
                cells.append(
                    CellBox(
                        row_start=r0,
                        col_start=c0,
                        row_end=r1,
                        col_end=c1,
                        center_row=(r0 + r1) / 2.0,
                        center_col=(c0 + c1) / 2.0,
                        fill_fraction=float(fill),
                    )
                )

    return cells


def _extract_cell(data: np.ndarray, cell: CellBox) -> np.ndarray:
    """Extract a cell's pixel data, NaN-filling if at edge."""
    return data[cell.row_start : cell.row_end, cell.col_start : cell.col_end].copy()


def _prepare_for_cv(patch: np.ndarray) -> np.ndarray:
    """Replace NaN with 0 and convert to float32 for OpenCV/skimage."""
    out = np.where(np.isnan(patch), 0.0, patch).astype(np.float32)
    # Normalize to zero-mean, unit-variance for better correlation
    valid = ~np.isnan(patch)
    if valid.sum() > 1:
        mean = out[valid].mean()
        std = out[valid].std()
        if std > 1e-10:
            out = (out - mean) / std
            out[~valid] = 0.0
    return out


# =====================================================================
# Stage 1: Coarse angular sweep
# =====================================================================


def _coarse_angular_sweep(
    ref_cell: np.ndarray,
    comp_map: np.ndarray,
    cell: CellBox,
    angle_min_deg: float,
    angle_max_deg: float,
    angle_step_deg: float,
) -> tuple[float, float, float, float]:
    """
    Find best angle + approximate translation via template matching.

    For each candidate angle, rotate the full comparison map and use
    normalised cross-correlation (``match_template``) to find where
    the reference cell best matches.

    :returns: ``(best_angle_deg, best_row, best_col, best_score)``
        where row/col are the matched centre in the original (unrotated)
        comparison map coordinates.
    """
    ref_patch = _prepare_for_cv(ref_cell)
    ph, pw = ref_patch.shape

    angles = np.arange(
        angle_min_deg, angle_max_deg + angle_step_deg / 2, angle_step_deg
    )

    best_score = -np.inf
    best_angle = 0.0
    best_row = cell.center_row
    best_col = cell.center_col

    comp_center = np.array([comp_map.shape[0] / 2.0, comp_map.shape[1] / 2.0])

    for angle in angles:
        # Rotate comparison map around its centre
        if abs(angle) > _MIN_ROTATION_DEG:
            rotated = _rotate_nan_safe(comp_map, angle)
        else:
            rotated = comp_map

        rotated_clean = np.where(np.isnan(rotated), 0.0, rotated).astype(np.float32)
        valid_rot = ~np.isnan(rotated)
        if valid_rot.sum() > 1:
            m = rotated_clean[valid_rot].mean()
            s = rotated_clean[valid_rot].std()
            if s > 1e-10:
                rotated_clean = (rotated_clean - m) / s
                rotated_clean[~valid_rot] = 0.0

        # Template matching — find best position for this cell
        if rotated_clean.shape[0] < ph or rotated_clean.shape[1] < pw:
            continue

        result = match_template(rotated_clean, ref_patch, pad_input=False)
        max_idx = np.unravel_index(np.argmax(result), result.shape)
        score = result[max_idx]

        if score > best_score:
            best_score = score
            best_angle = angle

            # Convert match position to centre coordinates in the rotated frame
            match_row = max_idx[0] + ph / 2.0
            match_col = max_idx[1] + pw / 2.0

            # Transform back to unrotated comparison map coordinates
            if abs(angle) > _MIN_ROTATION_DEG:
                angle_rad = np.radians(-angle)
                dr = match_row - comp_center[0]
                dc = match_col - comp_center[1]
                best_row = (
                    comp_center[0] + dr * np.cos(angle_rad) - dc * np.sin(angle_rad)
                )
                best_col = (
                    comp_center[1] + dr * np.sin(angle_rad) + dc * np.cos(angle_rad)
                )
            else:
                best_row = match_row
                best_col = match_col

    return best_angle, best_row, best_col, float(best_score)


# =====================================================================
# Stage 2: Sub-pixel translation refinement
# =====================================================================


def _subpixel_translation(
    ref_cell: np.ndarray,
    comp_map: np.ndarray,
    angle_deg: float,
    match_row: float,
    match_col: float,
) -> tuple[float, float]:
    """
    Refine translation to sub-pixel accuracy using phase cross-correlation.

    Extracts the comparison patch at the approximate match location from
    the rotated comparison map, then uses ``phase_cross_correlation`` to
    find the sub-pixel shift.

    :returns: Refined ``(row, col)`` centre of the match in the unrotated
        comparison map.
    """
    ph, pw = ref_cell.shape
    comp_center = np.array([comp_map.shape[0] / 2.0, comp_map.shape[1] / 2.0])

    # Rotate comparison map
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        rotated = _rotate_nan_safe(comp_map, angle_deg)
    else:
        rotated = comp_map.copy()

    # Extract patch at approximate location
    r0 = int(round(match_row - ph / 2.0))
    c0 = int(round(match_col - pw / 2.0))

    # Transform match_row/col to rotated frame
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        angle_rad = np.radians(angle_deg)
        dr = match_row - comp_center[0]
        dc = match_col - comp_center[1]
        rot_row = comp_center[0] + dr * np.cos(angle_rad) - dc * np.sin(angle_rad)
        rot_col = comp_center[1] + dr * np.sin(angle_rad) + dc * np.cos(angle_rad)
        r0 = int(round(rot_row - ph / 2.0))
        c0 = int(round(rot_col - pw / 2.0))

    # Clamp to valid range
    r0 = max(0, min(r0, rotated.shape[0] - ph))
    c0 = max(0, min(c0, rotated.shape[1] - pw))
    comp_patch = rotated[r0 : r0 + ph, c0 : c0 + pw]

    # Replace NaN for phase correlation
    ref_clean = np.where(np.isnan(ref_cell), 0.0, ref_cell)
    comp_clean = np.where(np.isnan(comp_patch), 0.0, comp_patch)

    try:
        shift, _, _ = phase_cross_correlation(
            ref_clean,
            comp_clean,
            upsample_factor=10,
        )
        # shift is [row_shift, col_shift]: comp needs to be shifted by this to match ref
        refined_rot_row = r0 + ph / 2.0 - shift[0]
        refined_rot_col = c0 + pw / 2.0 - shift[1]
    except Exception:
        refined_rot_row = r0 + ph / 2.0
        refined_rot_col = c0 + pw / 2.0

    # Transform back to unrotated coordinates
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        angle_rad = np.radians(-angle_deg)
        dr = refined_rot_row - comp_center[0]
        dc = refined_rot_col - comp_center[1]
        final_row = comp_center[0] + dr * np.cos(angle_rad) - dc * np.sin(angle_rad)
        final_col = comp_center[1] + dr * np.sin(angle_rad) + dc * np.cos(angle_rad)
    else:
        final_row = refined_rot_row
        final_col = refined_rot_col

    return float(final_row), float(final_col)


# =====================================================================
# Stage 3: ECC gradient refinement
# =====================================================================


def _ecc_refine(
    ref_cell: np.ndarray,
    comp_map: np.ndarray,
    angle_deg: float,
    match_row: float,
    match_col: float,
    angle_range_deg: float = 3.0,
    max_iter: int = 150,
    eps: float = 1e-4,
) -> tuple[float, float, float, float]:
    """
    Refine ``[dx, dy, θ]`` using OpenCV's ECC algorithm.

    :returns: ``(refined_angle_deg, refined_row, refined_col, ecc_score)``
        in unrotated comparison map coordinates.
    """
    ph, pw = ref_cell.shape
    comp_center = np.array([comp_map.shape[0] / 2.0, comp_map.shape[1] / 2.0])

    # Rotate comparison map to approximate angle
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        rotated = _rotate_nan_safe(comp_map, angle_deg)
    else:
        rotated = comp_map.copy()

    # Get match position in rotated frame
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        angle_rad = np.radians(angle_deg)
        dr = match_row - comp_center[0]
        dc = match_col - comp_center[1]
        rot_row = comp_center[0] + dr * np.cos(angle_rad) - dc * np.sin(angle_rad)
        rot_col = comp_center[1] + dr * np.sin(angle_rad) + dc * np.cos(angle_rad)
    else:
        rot_row, rot_col = match_row, match_col

    # Extract comparison patch
    r0 = int(round(rot_row - ph / 2.0))
    c0 = int(round(rot_col - pw / 2.0))
    r0 = max(0, min(r0, rotated.shape[0] - ph))
    c0 = max(0, min(c0, rotated.shape[1] - pw))
    comp_patch = rotated[r0 : r0 + ph, c0 : c0 + pw]

    # Prepare for ECC
    ref_f32 = _prepare_for_cv(ref_cell)
    comp_f32 = _prepare_for_cv(comp_patch)

    if ref_f32.shape != comp_f32.shape:
        return angle_deg, match_row, match_col, 0.0

    # Initial warp matrix: identity (small refinement around current estimate)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_f32,
            comp_f32,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )

        # Extract refined parameters
        # warp_matrix = [[cos θ, -sin θ, dx], [sin θ, cos θ, dy]]
        d_angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        dx = warp_matrix[0, 2]  # column shift
        dy = warp_matrix[1, 2]  # row shift

        # Refined position in rotated frame
        refined_rot_row = rot_row + dy
        refined_rot_col = rot_col + dx
        refined_angle_deg = angle_deg + np.degrees(d_angle_rad)

    except cv2.error:
        # ECC failed to converge — use input values
        refined_rot_row = rot_row
        refined_rot_col = rot_col
        refined_angle_deg = angle_deg

    # Transform back to unrotated coordinates
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        angle_rad_inv = np.radians(-angle_deg)
        dr = refined_rot_row - comp_center[0]
        dc = refined_rot_col - comp_center[1]
        final_row = (
            comp_center[0] + dr * np.cos(angle_rad_inv) - dc * np.sin(angle_rad_inv)
        )
        final_col = (
            comp_center[1] + dr * np.sin(angle_rad_inv) + dc * np.cos(angle_rad_inv)
        )
    else:
        final_row = refined_rot_row
        final_col = refined_rot_col

    # Compute ACCF at refined position
    accf = _compute_accf(ref_cell, comp_map, refined_angle_deg, final_row, final_col)

    return refined_angle_deg, float(final_row), float(final_col), accf


# =====================================================================
# ACCF computation
# =====================================================================


def _compute_accf(
    ref_cell: np.ndarray,
    comp_map: np.ndarray,
    angle_deg: float,
    center_row: float,
    center_col: float,
) -> float:
    """
    Compute the Area Cross-Correlation Function (ACCF) between a reference
    cell and a patch extracted from the comparison map at the given position
    and angle.
    """
    ph, pw = ref_cell.shape
    comp_center = np.array([comp_map.shape[0] / 2.0, comp_map.shape[1] / 2.0])

    # Rotate comparison map
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        rotated = _rotate_nan_safe(comp_map, angle_deg)
    else:
        rotated = comp_map

    # Get position in rotated frame
    if abs(angle_deg) > _MIN_ROTATION_DEG:
        angle_rad = np.radians(angle_deg)
        dr = center_row - comp_center[0]
        dc = center_col - comp_center[1]
        rot_row = comp_center[0] + dr * np.cos(angle_rad) - dc * np.sin(angle_rad)
        rot_col = comp_center[1] + dr * np.sin(angle_rad) + dc * np.cos(angle_rad)
    else:
        rot_row, rot_col = center_row, center_col

    # Extract patch
    r0 = int(round(rot_row - ph / 2.0))
    c0 = int(round(rot_col - pw / 2.0))
    r0 = max(0, min(r0, rotated.shape[0] - ph))
    c0 = max(0, min(c0, rotated.shape[1] - pw))
    comp_patch = rotated[r0 : r0 + ph, c0 : c0 + pw]

    if comp_patch.shape != ref_cell.shape:
        return np.nan

    # ACCF on common valid pixels
    both_valid = ~np.isnan(ref_cell) & ~np.isnan(comp_patch)
    n_valid = both_valid.sum()
    if n_valid < 10:
        return np.nan

    r = ref_cell[both_valid]
    c = comp_patch[both_valid]

    # Remove mean (leveling)
    r = r - r.mean()
    c = c - c.mean()

    denom = np.sqrt(np.sum(r**2) * np.sum(c**2))
    if denom < 1e-30:
        return np.nan

    return float(np.sum(r * c) / denom)


# =====================================================================
# Main pipeline
# =====================================================================


def _engine_register_cells(
    ref_leveled: np.ndarray,
    comp_leveled: np.ndarray,
    ref_processed: np.ndarray,
    comp_processed: np.ndarray,
    pixel_spacing: np.ndarray,
    cell_size_m: np.ndarray,
    grid_origin_px: tuple[int, int],
    angle_min_deg: float = -30.0,
    angle_max_deg: float = 30.0,
    angle_step_deg: float = 1.0,
    min_fill: float = 0.35,
) -> list[CellRegistrationResult]:
    """
    Register cells from the reference surface against the comparison surface.

    :param ref_leveled: Reference height map (leveled) for registration.
    :param comp_leveled: Comparison height map (leveled) for registration.
    :param ref_processed: Reference height map (filtered) for final ACCF.
    :param comp_processed: Comparison height map (filtered) for final ACCF.
    :param pixel_spacing: ``[row_spacing, col_spacing]`` in metres.
    :param cell_size_m: ``[row_size, col_size]`` in metres.
    :param grid_origin_px: ``(row_offset, col_offset)`` pixel offset for cell grid.
    :param angle_min_deg: Minimum search angle in degrees.
    :param angle_max_deg: Maximum search angle in degrees.
    :param angle_step_deg: Angular step for coarse sweep in degrees.
    :param min_fill: Minimum valid-pixel fill fraction per cell.
    :returns: List of per-cell results.
    """
    cell_size_px = np.round(cell_size_m / pixel_spacing).astype(int)

    # Generate cell grid
    cells = _generate_cells(ref_leveled, tuple(cell_size_px), min_fill)

    results = []
    for cell in cells:
        ref_cell_lev = _extract_cell(ref_leveled, cell)

        # --- Stage 1: Coarse angular sweep ---
        best_angle, match_row, match_col, coarse_score = _coarse_angular_sweep(
            ref_cell_lev,
            comp_leveled,
            cell,
            angle_min_deg,
            angle_max_deg,
            angle_step_deg,
        )

        if coarse_score < -0.5:
            results.append(
                CellRegistrationResult(
                    center_reference=np.array([cell.center_row, cell.center_col])
                    * pixel_spacing,
                    center_comparison=np.full(2, np.nan),
                    registration_angle=np.nan,
                    accf=np.nan,
                    fill_fraction=cell.fill_fraction,
                    is_valid=False,
                )
            )
            continue

        # --- Stage 2: Sub-pixel translation ---
        refined_row, refined_col = _subpixel_translation(
            ref_cell_lev,
            comp_leveled,
            best_angle,
            match_row,
            match_col,
        )

        # --- Stage 3: ECC refinement ---
        final_angle, final_row, final_col, reg_accf = _ecc_refine(
            ref_cell_lev,
            comp_leveled,
            best_angle,
            refined_row,
            refined_col,
        )

        # --- Stage 4: Final ACCF on processed data ---
        ref_cell_proc = _extract_cell(ref_processed, cell)
        final_accf = _compute_accf(
            ref_cell_proc,
            comp_processed,
            final_angle,
            final_row,
            final_col,
        )

        # Convert pixel positions to physical coordinates
        center_ref_m = np.array([cell.center_row, cell.center_col]) * pixel_spacing
        center_comp_m = np.array([final_row, final_col]) * pixel_spacing
        angle_rad = np.radians(final_angle)

        results.append(
            CellRegistrationResult(
                center_reference=center_ref_m,
                center_comparison=center_comp_m,
                registration_angle=angle_rad,
                accf=final_accf if not np.isnan(final_accf) else reg_accf,
                fill_fraction=cell.fill_fraction,
                is_valid=True,
            )
        )

    return results
