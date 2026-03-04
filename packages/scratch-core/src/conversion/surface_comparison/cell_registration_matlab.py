"""
cell_registration_matlab.py — Faithful Python translation of MATLAB cell_corr_analysis.

Key fix vs earlier versions:
  - cell_init: cells share parent map's vCenterG, adjust vCenterL by pixel offset
    (matching MATLAB: Cell.Map1.vCenterG = Map1.vCenterG,
     Cell.Map1.vCenterL = Map1.vCenterL - vTrans.*Map1.vPixSep)

Coordinate convention (MATLAB):
  - Positions: [row, col] in physical units (metres)
  - vCenterG: global position of the image centre
  - vCenterL: local centre position (adjusted for sub-images)
  - vPixSep: [row_spacing, col_spacing]
  - Pixel indexing: 1-based, column-major
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator


# =====================================================================
# Data classes
# =====================================================================


@dataclass
class MapStruct:
    """Mirrors MATLAB's Map structure."""

    map: np.ndarray
    vCenterG: np.ndarray
    vCenterL: np.ndarray
    angle: float
    vPixSep: np.ndarray


@dataclass
class CellResult:
    """Per-cell output matching MATLAB's Cell(i) structure."""

    vPos1: np.ndarray
    vPos2: np.ndarray
    dAngle: float
    accf: float
    fill1: float
    vdPos: np.ndarray
    bValid: bool = True


# =====================================================================
# Low-level helpers
# =====================================================================


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


def _map_interp2(interp, pix):
    return interp(pix)


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


def _similarity2cost(sim_val, sim_metric, to_cost):
    if sim_metric.lower() == "accf":
        return -sim_val if to_cost else -sim_val
    raise ValueError(f"Unsupported metric: {sim_metric}")


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


def _level_map(map_data, viParLevel, vCenterL, vPixSep):
    nrows, ncols = map_data.shape
    result = map_data.copy()
    rows = np.arange(1, nrows + 1) * vPixSep[0] - vCenterL[0]
    cols = np.arange(1, ncols + 1) * vPixSep[1] - vCenterL[1]
    cc, rr = np.meshgrid(cols, rows)
    valid = ~np.isnan(result)
    if valid.sum() < len(viParLevel) + 1:
        return result
    x, y, z = rr[valid], cc[valid], result[valid]
    cols_list = []
    for p in viParLevel:
        if p == 1:
            cols_list.append(np.ones_like(x))
        elif p == 2:
            cols_list.append(x)
        elif p == 3:
            cols_list.append(y)
    if cols_list:
        A = np.column_stack(cols_list)
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        Af = np.column_stack(
            [
                np.ones(nrows * ncols)
                if p == 1
                else rr.ravel()
                if p == 2
                else cc.ravel()
                for p in viParLevel
            ]
        )
        result = result - (Af @ coeffs).reshape(nrows, ncols)
    return result


# =====================================================================
# Cell initialisation — MATLAB cell_init.m faithful translation
# =====================================================================


def _cell_init(
    map1: MapStruct,
    vCellSize: np.ndarray,
    vCellPosition: np.ndarray,
    cellFillRefMin: float,
) -> list[dict]:
    """
    Generate cells on Map1.

    MATLAB key: Cell.Map1.vCenterG = Map1.vCenterG (shared!)
                Cell.Map1.vCenterL = Map1.vCenterL - vTrans * Map1.vPixSep
    where vTrans is the 0-based pixel offset of the sub-image origin.
    """
    vPixSep = map1.vPixSep
    nrows, ncols = map1.map.shape

    cell_px = np.round(vCellSize / vPixSep).astype(int)

    # Generate cell centres (local coordinates in Map1 frame)
    # First get global positions of all valid pixels
    vi1 = np.flatnonzero(~np.isnan(map1.map.ravel(order="F"))) + 1
    mp1 = _trans_ind2sub(map1.map.shape, vi1)
    mp1_pos = _pix2pos(mp1, map1.vCenterG, map1.angle, map1.vCenterL, map1.vPixSep)

    # Generate cell grid positions
    map_extent_r = nrows * vPixSep[0]
    map_extent_c = ncols * vPixSep[1]

    centres_row = []
    r = vCellPosition[0]
    while r - vCellSize[0] / 2 > vPixSep[0]:
        r -= vCellSize[0]
    while r - vCellSize[0] / 2 < map_extent_r:
        centres_row.append(r)
        r += vCellSize[0]

    centres_col = []
    c = vCellPosition[1]
    while c - vCellSize[1] / 2 > vPixSep[1]:
        c -= vCellSize[1]
    while c - vCellSize[1] / 2 < map_extent_c:
        centres_col.append(c)
        c += vCellSize[1]

    cells = []

    for cr in centres_row:
        for cc in centres_col:
            # Cell centre in global coordinates
            # MATLAB: mCellPosition(ic,:) which is in global frame after cell_generate
            # For Map1 with angle=0: vPos1 = [cr, cc] (local = global)
            vPos1 = map1.vCenterG + _rotz(
                np.array([cr, cc]) - map1.vCenterL,
                map1.angle,
            )

            # Identify pixels inside cell (rectangle)
            half_r = vCellSize[0] / 2
            half_c = vCellSize[1] / 2
            vb1 = (np.abs(mp1_pos[:, 0] - vPos1[0]) <= half_r + 1e-10) & (
                np.abs(mp1_pos[:, 1] - vPos1[1]) <= half_c + 1e-10
            )

            nPoint1 = int(np.sum(vb1))
            nPointCellRefMin = max(
                9,
                int(
                    np.floor(
                        cellFillRefMin * vCellSize[0] * vCellSize[1] / np.prod(vPixSep)
                    )
                ),
            )
            if nPoint1 < nPointCellRefMin:
                continue

            # Extract sub-image: find bounding box of selected pixels
            # MATLAB: map2map_sub_ind returns (sub_map, vTrans)
            # vTrans is the 0-based [row, col] offset of the sub-image
            sel_vi = vi1[vb1]  # 1-based column-major indices
            sel_rows = (sel_vi - 1) % nrows
            sel_cols = (sel_vi - 1) // nrows
            r_min, r_max = sel_rows.min(), sel_rows.max()
            c_min, c_max = sel_cols.min(), sel_cols.max()

            # vTrans: 0-based offset (matches MATLAB map2map_sub_ind)
            vTrans = np.array([r_min, c_min], dtype=np.float64)

            sub = map1.map[r_min : r_max + 1, c_min : c_max + 1].copy()

            # Fill fraction
            fill1 = float(nPoint1) / float(cell_px[0] * cell_px[1])

            # MATLAB: Cell.Map1.vCenterG = Map1.vCenterG (SHARED!)
            #         Cell.Map1.vCenterL = Map1.vCenterL - vTrans .* Map1.vPixSep
            sub_vCenterG = map1.vCenterG.copy()
            sub_vCenterL = map1.vCenterL - vTrans * map1.vPixSep

            cell_map = MapStruct(
                map=sub,
                vCenterG=sub_vCenterG,
                vCenterL=sub_vCenterL,
                angle=map1.angle,
                vPixSep=vPixSep.copy(),
            )

            cells.append(
                {
                    "Map1": cell_map,
                    "vPos1": vPos1.copy(),
                    "nPoint1": nPoint1,
                    "fill1": fill1,
                }
            )

    return cells


# =====================================================================
# Evaluate similarity at a given angle
# =====================================================================


def _evaluate_similarity(
    cell_map: MapStruct,
    map2: MapStruct,
    map2_interp,
    angle: float,
    viParLevel: np.ndarray,
    n_overlap_min: int,
    scale: float = 1.0,
) -> float:
    """Evaluate ACCF between a cell and Map2 at a given angle."""
    map1 = cell_map.map
    vi1 = np.flatnonzero(~np.isnan(map1.ravel(order="F"))) + 1
    if len(vi1) < n_overlap_min:
        return np.nan
    mp1 = _trans_ind2sub(map1.shape, vi1)
    mp1 = _pix2pos(
        mp1, cell_map.vCenterG, cell_map.angle, cell_map.vCenterL, cell_map.vPixSep
    )
    mp2 = _pos2pix(
        mp1, map2.vCenterG, angle, map2.vCenterL * scale, map2.vPixSep * scale
    )
    vm2 = _map_interp2(map2_interp, mp2)
    vb = ~np.isnan(vm2)
    n_valid = np.sum(vb)
    if n_valid < n_overlap_min:
        return np.nan
    vm1 = map1.ravel(order="F")[vi1[vb] - 1]
    mm = np.column_stack([vm1, vm2[vb]])
    mm = _level_2d_pos(mp1[vb, 0], mp1[vb, 1], mm, viParLevel)
    s1, s2 = mm[:, 0], mm[:, 1]
    denom = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    if denom < 1e-30:
        return np.nan
    return float(np.sum(s1 * s2) / denom)


# =====================================================================
# Coarse angular sweep
# =====================================================================


def _cell_corr_angle(
    cells,
    map2,
    vAngles,
    viParLevel,
    n_overlap_min_per_cell,
    scale=1.0,
    verbose=False,
):
    """Evaluate ACCF for all cells at all angles."""
    map2_interp = _build_interpolator(map2.map, method="cubic")
    n_cells = len(cells)
    sim_vals = [np.full(len(vAngles), np.nan) for _ in range(n_cells)]
    for ia, angle in enumerate(vAngles):
        for ic, cell in enumerate(cells):
            sim_vals[ic][ia] = _evaluate_similarity(
                cell["Map1"],
                map2,
                map2_interp,
                angle,
                viParLevel,
                n_overlap_min_per_cell[ic],
                scale,
            )
    if verbose:
        for ic in range(n_cells):
            valid = ~np.isnan(sim_vals[ic])
            if valid.any():
                best_idx = np.nanargmax(sim_vals[ic])
                print(
                    f"  Coarse cell {ic}: best angle={np.degrees(vAngles[best_idx]):.3f}°, "
                    f"ACCF={sim_vals[ic][best_idx]:.4f}"
                )
    return sim_vals


# =====================================================================
# Gradient-based fine registration
# =====================================================================


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
    for idx, p in enumerate(vip):
        if p == 0:
            mG[:, idx] = m_grad[:, 0]
        elif p == 1:
            mG[:, idx] = m_grad[:, 1]
        elif p == 2:
            mG[:, idx] = -m_grad[:, 0] * mp1_v[:, 1] + m_grad[:, 1] * mp1_v[:, 0]
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

    vdPar, _, _, _ = np.linalg.lstsq(mG, lam * s1 - s2, rcond=None)
    vPar_out = vPar.copy()
    for idx, p in enumerate(vip):
        vPar_out[p] -= vdPar[idx]
    vPar_out = np.maximum(vPar_out, m_range[:, 0])
    vPar_out = np.minimum(vPar_out, m_range[:, 1])
    return cost_val, True, vPar_out


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
            done = np.all(np.abs(vPar_next - vPar) <= tol_x) or cost_new > (
                cost_val - conv_sim
            )
            if cost_new < cost_val:
                vPar = vPar_next.copy()
                cost_val = cost_new

    if not b_valid:
        return NAN_RESULT
    sim_val = _similarity2cost(cost_val, "accf", False)
    return vPar[:2].copy(), float(vPar[2]), float(vPar[3]), float(sim_val)


# =====================================================================
# Main: cell_corr_analysis
# =====================================================================


def cell_corr_analysis(
    map1: MapStruct,
    map2: MapStruct,
    vCellSize: np.ndarray,
    vCellPosition: np.ndarray,
    shiftAngleMin: float,
    shiftAngleMax: float,
    cellFillRefMin: float = 0.35,
    cellFillRegMin: float = 0.25,
    cellFillRedMax: float = 0.50,
    viParLevel: np.ndarray = np.array([1]),
    viParLevelReg: np.ndarray | None = None,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    cellConvAnglePix: float = 0.5,
    nCellRegImageReductionMax: int = 4,
    nInterval: int = 3,
    bEval180: bool = False,
    verbose: bool = False,
) -> list[CellResult]:
    if map1.angle != 0.0:
        raise ValueError("Angle of reference map must equal 0")

    vCenterG2Init = map2.vCenterG.copy()
    vCenterL2Init = map2.vCenterL.copy()
    angle2Init = map2.angle

    angleMin = map2.angle + shiftAngleMin
    angleMax = map2.angle + shiftAngleMax
    bAngularReg = (shiftAngleMax - shiftAngleMin) > 1e-10

    # Generate cells on Map1
    cells = _cell_init(map1, vCellSize, vCellPosition, cellFillRefMin)
    if not cells:
        return []

    cellArea = vCellSize[0] * vCellSize[1]
    nPointCellRegMin = int(np.floor(cellFillRegMin * cellArea / np.prod(map1.vPixSep)))

    # ================================================================
    # Coarse angular sweep (on full-resolution maps, no decimation)
    # ================================================================
    coarse_best_angles = None
    dAngle = 0.0

    if bAngularReg:
        # Angular resolution from cell size in pixels
        cell_px = vCellSize / map1.vPixSep
        radiusP = 0.5 * np.sqrt(np.sum(cell_px**2))
        dAngle = cellConvAnglePix / radiusP

        # Build angle list
        vAngles = np.arange(angleMin, angleMax + 1e-6, dAngle)
        vAngles = vAngles - np.mean(vAngles) + 0.5 * (angleMin + angleMax)

        if bEval180:
            vAngles = np.concatenate([vAngles, vAngles + np.pi])

        if verbose:
            print(
                f"  Coarse sweep: {len(vAngles)} angles, dAngle={np.degrees(dAngle):.3f}°"
            )

        # Overlap minimums per cell
        n_overlap_mins = []
        for cell in cells:
            nom = max(
                nPointCellRegMin, int(np.floor(cell["nPoint1"] * (1 - cellFillRedMax)))
            )
            n_overlap_mins.append(int(1.02 * nom))

        # Perform angular sweep on full-resolution maps
        sim_vals = _cell_corr_angle(
            cells,
            map2,
            vAngles,
            viParLevel,
            n_overlap_mins,
            1.0,
            verbose,
        )

        # Best angle per cell
        coarse_best_angles = []
        for sv in sim_vals:
            costs = np.array(
                [
                    _similarity2cost(v, "accf", True) if not np.isnan(v) else np.inf
                    for v in sv
                ]
            )
            coarse_best_angles.append(vAngles[np.argmin(costs)])

    # ================================================================
    # Fine registration per cell
    # ================================================================
    results = []

    for ic, cell in enumerate(cells):
        cell_map = cell["Map1"]

        if viParLevelReg is not None and len(viParLevelReg) > 0:
            cell_map.map = _level_map(
                cell_map.map,
                viParLevelReg,
                cell_map.vCenterL,
                cell_map.vPixSep,
            )

        if not np.isnan(cellFillRedMax):
            n_overlap_min = max(
                nPointCellRegMin,
                int(np.floor(cell["nPoint1"] * (1 - cellFillRedMax))),
            )
        else:
            n_overlap_min = nPointCellRegMin

        if bAngularReg and coarse_best_angles is not None:
            best_angle = (
                coarse_best_angles[ic]
                if ic < len(coarse_best_angles)
                else 0.5 * (angleMin + angleMax)
            )
            map2_angle = best_angle
            fine_shiftMin = max(angleMin - map2_angle, -nInterval * dAngle)
            fine_shiftMax = min(angleMax - map2_angle, nInterval * dAngle)
            fine_angleMin = map2_angle + fine_shiftMin
            fine_angleMax = map2_angle + fine_shiftMax
        else:
            map2_angle = 0.5 * (angleMin + angleMax)
            fine_angleMin = map2_angle
            fine_angleMax = map2_angle

        map2_reg = MapStruct(
            map=map2.map,
            vCenterG=map2.vCenterG.copy(),
            vCenterL=map2.vCenterL.copy(),
            angle=map2_angle,
            vPixSep=map2.vPixSep.copy(),
        )

        vCG2, angle2, scale2, sim_val = maps_register_fine_gradient(
            cell_map=cell_map,
            comp_map=map2_reg,
            angle_min=fine_angleMin,
            angle_max=fine_angleMax,
            n_overlap_min=n_overlap_min,
            viParLevel=viParLevel,
            n_conv_iter=150,
            conv_sim=1e-4,
            conv_pos=0.1,
            conv_angle=1e-3,
            conv_scale=1e-3,
            scale_min=scale_min,
            scale_max=scale_max,
            verbose=verbose,
        )

        if np.isnan(vCG2[0]):
            results.append(
                CellResult(
                    vPos1=cell["vPos1"],
                    vPos2=np.full(2, np.nan),
                    dAngle=np.nan,
                    accf=np.nan,
                    fill1=cell["fill1"],
                    vdPos=np.full(2, np.nan),
                    bValid=False,
                )
            )
            continue

        # vPos2 computation (MATLAB lines 316-328)
        Map2_vCenterL_scaled = vCenterL2Init * scale2
        vPos2 = _rotz(cell["vPos1"] - vCG2, -angle2) + Map2_vCenterL_scaled
        vPos2 = _rotz(vPos2 - vCenterL2Init, angle2Init) + vCenterG2Init

        dAngle_val = angle2 - angle2Init
        vdPos = vCG2 - vCenterG2Init

        results.append(
            CellResult(
                vPos1=cell["vPos1"],
                vPos2=vPos2,
                dAngle=dAngle_val,
                accf=sim_val,
                fill1=cell["fill1"],
                vdPos=vdPos,
                bValid=True,
            )
        )

        if verbose:
            print(
                f"  Cell {ic}: fill={cell['fill1']:.2f}  "
                f"dX={vdPos[0] * 1e6:.1f}  dY={vdPos[1] * 1e6:.1f}  "
                f"dA={np.degrees(dAngle_val):.3f}°  ACCF={sim_val:.4f}"
            )

    return results
