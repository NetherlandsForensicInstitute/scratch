"""
Per-cell registration: two-stage pipeline.

Stage 1  (coarse angular sweep)
    The comparison image is rotated once per candidate angle.  For each
    rotation, every reference cell is matched against the rotated image using
    normalised cross-correlation (NCC).  NCC preserves frequency amplitudes,
    so its peak value is a true similarity score in [-1, 1].  This makes it
    possible to compare scores across different angles and reliably pick the
    best rotation for each cell.

Stage 2  (gradient fine registration)
    MATLAB-faithful gradient-based fine registration (``maps_register_fine_gradient``).
    Jointly refines translation and rotation by iteratively solving a linearised
    gradient system in physical coordinates on the *full* comparison image — no
    crop, no fixed window.  Seeded directly from the Stage 1 NCC position and
    angle estimate.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.feature import match_template
from skimage.transform import rotate

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    Cell,
)
from conversion.surface_comparison.grid import (
    _find_grid_origin,
    generate_grid_centers,
)
from conversion.surface_comparison.utils import (
    meters_to_pixels,
    compute_top_left_pixel_of_cell,
)


# Fine registration: ±3° angular search window around NCC seed (matches MATLAB default)
_FINE_ANGLE_HALF_RAD = np.radians(3.0)
# Minimum cell fill fraction for fine registration overlap check
_FINE_CELL_FILL_REG_MIN = 0.25


@dataclass
class MapStruct:
    """Mirrors MATLAB's Map structure."""

    map: np.ndarray
    vCenterG: np.ndarray
    vCenterL: np.ndarray
    angle: float
    vPixSep: np.ndarray


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
    :returns: CellResult list for all cells that pass the fill-fraction check.
    """
    origin = _find_grid_origin(reference_image, params)
    centers = generate_grid_centers(reference_image, origin, params)

    # Pre-extract reference cell_dataes and compute fill fractions.
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

        cell = Cell(
            center_reference=center,
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

    # ---- Stage 2: gradient fine registration per cell ----
    for cell in valid_cells:
        _fine_tune_cell(cell, reference_image, comparison_image, params)

    return valid_cells


def maps_register_fine_gradient(
    cell_map,
    comp_map,
    angle_min,
    angle_max,
    n_overlap_min,
    viParLevel=np.array([1]),
    n_conv_iter=150,
    conv_sim=1e-4,
    conv_pos=0.1,
    conv_angle=1e-3,
    conv_scale=1e-3,
    scale_min=1.0,
    scale_max=1.0,
    pos_min=None,
    pos_max=None,
    verbose=False,
):
    NAN_RESULT = (np.full(2, np.nan), np.nan, np.nan, np.nan)
    if pos_min is None:
        pos_min = np.array([np.nan, np.nan])
    if pos_max is None:
        pos_max = np.array([np.nan, np.nan])

    scale0 = (scale_min + scale_max) / 2.0
    vPar = np.array(
        [comp_map.vCenterG[0], comp_map.vCenterG[1], comp_map.angle, scale0]
    )

    m_range = np.array(
        [
            [pos_min[0], pos_max[0]],
            [pos_min[1], pos_max[1]],
            [angle_min, angle_max],
            [scale_min, scale_max],
        ]
    )
    m_range[np.isnan(m_range[:, 0]), 0] = -np.inf
    m_range[np.isnan(m_range[:, 1]), 1] = np.inf
    vPar = np.maximum(vPar, m_range[:, 0])
    vPar = np.minimum(vPar, m_range[:, 1])

    vi1 = np.flatnonzero(~np.isnan(cell_map.map.ravel(order="F"))) + 1
    if len(vi1) < n_overlap_min:
        return NAN_RESULT
    mp1 = _trans_ind2sub(cell_map.map.shape, vi1)
    mp1 = _pix2pos(
        mp1, cell_map.vCenterG, cell_map.angle, cell_map.vCenterL, cell_map.vPixSep
    )
    interp = _build_interpolator(comp_map.map, method="cubic")
    tol_x = np.array([conv_pos, conv_pos, conv_angle, conv_scale])

    cost_val, b_valid, vPar_new = _fun_minimize_gradient(
        vPar,
        cell_map.map,
        comp_map.vCenterL,
        comp_map.vPixSep,
        vi1.copy(),
        mp1.copy(),
        interp,
        viParLevel,
        n_overlap_min,
        m_range,
        cell_map.vPixSep,
        verbose,
    )
    if not b_valid:
        return NAN_RESULT

    n_evals = 1
    done = False
    while n_evals <= n_conv_iter and not done:
        cost_new, b_valid_new, vPar_next = _fun_minimize_gradient(
            vPar_new,
            cell_map.map,
            comp_map.vCenterL,
            comp_map.vPixSep,
            vi1.copy(),
            mp1.copy(),
            interp,
            viParLevel,
            n_overlap_min,
            m_range,
            cell_map.vPixSep,
            verbose,
        )
        n_evals += 1
        if not b_valid_new:
            b_valid = True
            done = True
        else:
            done = np.all(np.abs(vPar_next - vPar_new) <= tol_x) or cost_new > (
                cost_val - conv_sim
            )
            if cost_new < cost_val:
                vPar = vPar_next.copy()
                cost_val = cost_new
            vPar_new = vPar_next.copy()

    if not b_valid:
        return NAN_RESULT
    sim_val = _similarity2cost(cost_val, "accf", False)
    return vPar[:2].copy(), float(vPar[2]), float(vPar[3]), float(sim_val)


def _similarity2cost(sim_val, sim_metric, to_cost):
    if sim_metric.lower() == "accf":
        return -sim_val if to_cost else -sim_val
    raise ValueError(f"Unsupported metric: {sim_metric}")


def _build_interpolator(map2, method="cubic"):
    nrows, ncols = map2.shape
    row_coords = np.arange(1, nrows + 1, dtype=np.float64)
    col_coords = np.arange(1, ncols + 1, dtype=np.float64)
    map_clean = np.where(np.isnan(map2), 0.0, map2)
    return RegularGridInterpolator(
        (row_coords, col_coords),
        map_clean,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )


def _trans_ind2sub(shape, vi):
    nrows = shape[0]
    row1 = (vi - 1) % nrows + 1
    col1 = (vi - 1) // nrows + 1
    return np.column_stack([row1, col1]).astype(np.float64)


def _pix2pos(pix, vCenterG, angle, vCenterL, vPixSep):
    """1-based [row,col] pixels → global [row,col] positions."""
    scaled = pix * vPixSep
    centred = scaled - vCenterL
    if abs(angle) > 1e-15:
        c, s = np.cos(angle), np.sin(angle)
        rotated = np.column_stack(
            [
                c * centred[:, 0] - s * centred[:, 1],
                s * centred[:, 0] + c * centred[:, 1],
            ]
        )
    else:
        rotated = centred
    return rotated + vCenterG


def _fun_minimize_gradient(
    vPar,
    map1,
    map2_vCenterL,
    map2_vPixSep,
    vi1,
    mp1,
    interp,
    viParLevel,
    n_overlap_min,
    m_range,
    map1_vPixSep,
    verbose=False,
):
    scale = vPar[3]
    mp2 = _pos2pix(mp1, vPar[:2], vPar[2], map2_vCenterL * scale, map2_vPixSep * scale)
    vm2 = _map_interp2(interp, mp2)
    vb = ~np.isnan(vm2)
    n_valid = np.sum(vb)
    if n_valid < n_overlap_min:
        return 1.0 + 0.01 * (n_overlap_min - n_valid), False, vPar.copy()
    vi1_v, mp1_v, vm2_v = vi1[vb], mp1[vb], vm2[vb]
    vm1 = map1.ravel(order="F")[vi1_v - 1]
    mm = np.column_stack([vm1, vm2_v])
    mm = _level_2d_pos(mp1_v[:, 0], mp1_v[:, 1], mm, viParLevel)
    s1, s2 = mm[:, 0], mm[:, 1]
    denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    if denom < 1e-30:
        return 1.0, False, vPar.copy()
    sim_val = np.sum(s1 * s2) / denom
    cost_val = _similarity2cost(sim_val, "accf", True)
    if verbose:
        print(f"    Points: {n_valid}  ACCF: {sim_val:.6f}")

    # ECC gradient update
    vip = np.where(m_range[:, 1] > m_range[:, 0])[0]
    if len(vip) == 0:
        return cost_val, True, vPar.copy()
    sub_shape, offset, vi2 = _trans_ind2size(map1.shape, vi1_v)
    sub_nrows = sub_shape[0]
    map_d = np.full(sub_shape, np.nan)
    vi2_row0 = (vi2 - 1) % sub_nrows
    vi2_col0 = (vi2 - 1) // sub_nrows
    map_d[vi2_row0, vi2_col0] = mm[:, 1]
    grad_row, grad_col = _map_gradient(map_d, map1_vPixSep)
    m_grad = np.column_stack(
        [
            grad_row[vi2_row0, vi2_col0],
            grad_col[vi2_row0, vi2_col0],
        ]
    )
    vb_grad = np.all(~np.isnan(m_grad), axis=1)
    m_grad, mm, mp1_v = m_grad[vb_grad], mm[vb_grad], mp1_v[vb_grad]
    if len(m_grad) < max(len(vip) + 1, 4):
        return cost_val, True, vPar.copy()

    n_pts = len(m_grad)
    mG = np.zeros((n_pts, len(vip)))
    mp1_rel = mp1_v - vPar[:2]
    for idx, p in enumerate(vip):
        if p == 0:
            mG[:, idx] = -m_grad[:, 0]
        elif p == 1:
            mG[:, idx] = -m_grad[:, 1]
        elif p == 2:
            mG[:, idx] = m_grad[:, 0] * mp1_rel[:, 1] - m_grad[:, 1] * mp1_rel[:, 0]
        elif p == 3:
            mG[:, idx] = m_grad[:, 0] * mp1_v[:, 0] + m_grad[:, 1] * mp1_v[:, 1]
    mG = mG - mG.mean(axis=0)

    s1, s2 = mm[:, 0], mm[:, 1]
    vp1, _, _, _ = np.linalg.lstsq(mG, s1, rcond=None)
    vp2, _, _, _ = np.linalg.lstsq(mG, s2, rcond=None)

    s1_s2 = s1 @ s2
    s1_G_vp2 = s1 @ (mG @ vp2)
    if s1_s2 > s1_G_vp2:
        numer = s2 @ s2 - s2 @ (mG @ vp2)
        denom_lam = s1_s2 - s1_G_vp2
        if abs(denom_lam) < 1e-30:
            return cost_val, True, vPar.copy()
        lam = numer / denom_lam
    else:
        s2_G_vp2 = s2 @ (mG @ vp2)
        s1_G_vp1 = s1 @ (mG @ vp1)
        if abs(s1_G_vp1) < 1e-30:
            return cost_val, True, vPar.copy()
        lam1 = np.sqrt(abs(s2_G_vp2 / s1_G_vp1))
        lam2 = (s1_G_vp2 - s1_s2) / s1_G_vp1
        lam = max(lam1, lam2)

    vdPar, _, _, _ = np.linalg.lstsq(mG, s2 - lam * s1, rcond=None)
    vPar_out = vPar.copy()
    for idx, p in enumerate(vip):
        vPar_out[p] -= vdPar[idx]
    vPar_out = np.maximum(vPar_out, m_range[:, 0])
    vPar_out = np.minimum(vPar_out, m_range[:, 1])
    return cost_val, True, vPar_out


def _map_gradient(map_data, vPixSep):
    grad_row = np.gradient(map_data, vPixSep[0], axis=0)
    grad_col = np.gradient(map_data, vPixSep[1], axis=1)
    return grad_row, grad_col


def _trans_ind2size(full_shape, vi1):
    nrows = full_shape[0]
    row0 = (vi1 - 1) % nrows
    col0 = (vi1 - 1) // nrows
    r_min, r_max = row0.min(), row0.max()
    c_min, c_max = col0.min(), col0.max()
    sub_nrows = r_max - r_min + 1
    sub_ncols = c_max - c_min + 1
    sub_row0 = row0 - r_min
    sub_col0 = col0 - c_min
    vi2 = sub_row0 + sub_col0 * sub_nrows + 1
    return (sub_nrows, sub_ncols), (r_min, c_min), vi2


def _map_interp2(interp, pix):
    return interp(pix)


def _pos2pix(pos, vCenterG, angle, vCenterL, vPixSep):
    """Global [row,col] positions → 1-based [row,col] pixels."""
    shifted = pos - vCenterG
    if abs(angle) > 1e-15:
        c, s = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack(
            [
                c * shifted[:, 0] - s * shifted[:, 1],
                s * shifted[:, 0] + c * shifted[:, 1],
            ]
        )
    else:
        rotated = shifted
    return (rotated + vCenterL) / vPixSep


def _level_2d_pos(x, y, mm, viParLevel):
    if viParLevel is None or len(viParLevel) == 0:
        return mm
    n = len(x)
    result = mm.copy()
    cols = []
    for p in viParLevel:
        if p == 1:
            cols.append(np.ones(n))
        elif p == 2:
            cols.append(x)
        elif p == 3:
            cols.append(y)
        elif p == 4:
            cols.append(x * y)
        elif p == 5:
            cols.append(x**2)
        elif p == 6:
            cols.append(y**2)
    if not cols:
        return mm
    A = np.column_stack(cols)
    for col_idx in range(mm.shape[1]):
        data = mm[:, col_idx]
        vb = ~np.isnan(data)
        if vb.sum() < len(cols) + 1:
            continue
        coeffs, _, _, _ = np.linalg.lstsq(A[vb], data[vb], rcond=None)
        result[:, col_idx] = data - A @ coeffs
    return result


def _rotz(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    if v.ndim == 1:
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return np.column_stack(
        [
            c * v[:, 0] - s * v[:, 1],
            s * v[:, 0] + c * v[:, 1],
        ]
    )


def _get_optimal_crosscorr_and_comparison_center(
    rotated: FloatArray2D, cell_data: FloatArray2D
) -> tuple[float, int, int]:
    mean_subtracted_rotated = _replace_nan_with_image_mean(rotated)
    mean_subtracted_cell_data = _replace_nan_with_image_mean(cell_data)
    """Compute optimal cross correlation and corresponding location of top-left idxs of mean_subtracted_cell_data.
        Mean_subtracted_rotated is reference.

    :param rotated: the rotated reference imag.
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


def _fine_tune_cell(
    cell: Cell,
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
) -> None:
    """
    Refine a single cell registration using the MATLAB-faithful gradient fine
    registration algorithm (``maps_register_fine_gradient``).

    Works in physical coordinates on the full comparison image — no crop.
    Seeded from the Stage 1 NCC position and angle estimate.

    Coordinate conventions
    ----------------------
    Application layer (ScanImage, Cell):
        ``[x, y]`` in metres, x = column direction, y = row direction.
        Pixel positions are 0-based: ``pos = 0based_pixel * pixel_spacing``.

    MATLAB engine (MapStruct):
        ``[row, col]`` in metres.  Pixel indexing is 1-based:
        ``pos = 1based_pixel * vPixSep`` (when ``vCenterG == vCenterL``).

    The conversion between the two systems is a shift of exactly one pixel:
        ``1based_pixel = 0based_pixel + 1``
        ``matlab_pos   = app_pos + vPixSep``

    :param cell: Cell instance with Stage 1 NCC result in
        ``center_comparison`` and ``angle_reference``.
    :param reference_image: Fixed surface map.
    :param comparison_image: Moving surface map.
    :param params: Algorithm parameters (used for cell size and fill fraction).
    """
    ps = reference_image.pixel_spacing  # [dx, dy]  (x=col, y=row)
    vPS = np.array([ps[1], ps[0]])  # [dy, dx]  — MATLAB [row, col]

    ref_nrows, ref_ncols = reference_image.data.shape
    comp_nrows, comp_ncols = comparison_image.data.shape

    # Global image centres; for full images vCenterG == vCenterL
    ref_vCG = np.array(
        [np.ceil(ref_nrows / 2) * vPS[0], np.ceil(ref_ncols / 2) * vPS[1]]
    )
    comp_vCG_init = np.array(
        [np.ceil(comp_nrows / 2) * vPS[0], np.ceil(comp_ncols / 2) * vPS[1]]
    )

    # ---- Recover the top-left pixel offset of this cell in the reference image ----
    # Reproduces the same arithmetic used in register_cells when the cell was extracted.
    cell_height, cell_width = cell.cell_data.shape
    cell_size_px = meters_to_pixels(params.cell_size, ps)
    top_row = max(0, int(round(cell.center_reference[1] / ps[1] - cell_size_px[1] / 2)))
    left_col = max(
        0, int(round(cell.center_reference[0] / ps[0] - cell_size_px[0] / 2))
    )
    vTrans = np.array([top_row, left_col], dtype=np.float64)

    # ---- Build cell MapStruct ----
    # MATLAB: Cell.Map1.vCenterG = Map1.vCenterG  (shared)
    #         Cell.Map1.vCenterL = Map1.vCenterL - vTrans * vPixSep
    cell_map = MapStruct(
        map=cell.cell_data.copy(),
        vCenterG=ref_vCG.copy(),
        vCenterL=ref_vCG - vTrans * vPS,
        angle=0.0,
        vPixSep=vPS.copy(),
    )

    # ---- Compute vPos1: cell centre in reference, MATLAB 1-based convention ----
    # 1-based centre pixel = (0-based top-left) + (cell_size - 1)/2 + 1
    vPos1 = np.array(
        [
            (top_row + (cell_height - 1) / 2.0 + 1) * vPS[0],
            (left_col + (cell_width - 1) / 2.0 + 1) * vPS[1],
        ]
    )

    # ---- Seed comp MapStruct from Stage 1 NCC result ----
    # NCC center_comparison is in 0-based [x, y] metres.
    # Convert to MATLAB 1-based [row, col] metres by adding one pixel per axis.
    angle_seed = np.radians(cell.angle_reference)
    comp_pos_m = np.array(
        [
            cell.center_comparison[1] + vPS[0],  # row (y direction)
            cell.center_comparison[0] + vPS[1],  # col (x direction)
        ]
    )
    # Derive vCenterG2 seed so that the coordinate transform maps vPos1 → comp_pos_m:
    #   vPos2 = rotz(vPos1 - vCG2, -angle2) + vCL2
    #   => vCG2 = vPos1 - rotz(vPos2 - vCL2, angle2)
    vCG2_seed = vPos1 - _rotz(comp_pos_m - comp_vCG_init, angle_seed)

    comp_map = MapStruct(
        map=comparison_image.data.copy(),
        vCenterG=vCG2_seed,
        vCenterL=comp_vCG_init.copy(),
        angle=angle_seed,
        vPixSep=vPS.copy(),
    )

    # ---- Minimum overlap for fine registration ----
    cell_area = params.cell_size[0] * params.cell_size[1]
    n_overlap_min = int(np.floor(_FINE_CELL_FILL_REG_MIN * cell_area / np.prod(vPS)))

    # ---- Run gradient fine registration ----
    vCG2_out, angle2_out, _, sim_val = maps_register_fine_gradient(
        cell_map=cell_map,
        comp_map=comp_map,
        angle_min=angle_seed - _FINE_ANGLE_HALF_RAD,
        angle_max=angle_seed + _FINE_ANGLE_HALF_RAD,
        n_overlap_min=n_overlap_min,
        viParLevel=np.array([1]),
    )

    if np.isnan(sim_val):
        cell.best_score = 0.0
        return

    # ---- Convert result back to application [x, y] 0-based convention ----
    # Recover vPos2 (cell centre in comp image, MATLAB 1-based metres):
    #   vPos2 = rotz(vPos1 - vCG2, -angle2) + vCL2
    vPos2 = _rotz(vPos1 - vCG2_out, -angle2_out) + comp_vCG_init

    # Convert from 1-based metres to 0-based metres (subtract one pixel per axis)
    cell.center_comparison = np.array(
        [
            vPos2[1] - vPS[1],  # x = col direction
            vPos2[0] - vPS[0],  # y = row direction
        ]
    )
    cell.angle_reference = float(np.degrees(angle2_out))
    cell.best_score = float(sim_val)


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


def _replace_nan_with_image_mean(image: FloatArray2D) -> FloatArray2D:
    """Replace NaN with the image mean (or 0 if all-NaN)."""
    if not np.any(np.isnan(image)):
        return image.astype(np.float64)
    mean = float(np.nanmean(image)) if not np.all(np.isnan(image)) else 0.0
    return np.where(np.isnan(image), mean, image).astype(np.float64)
