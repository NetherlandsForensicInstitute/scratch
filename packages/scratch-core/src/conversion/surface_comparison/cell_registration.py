"""
Cell registration adapter.

Converts between the application's ``ScanImage`` / ``ComparisonParams`` /
``Cell`` types and the MATLAB-faithful ``cell_corr_analysis`` engine.

Coordinate conventions
----------------------
Application (ScanImage, Cell):
    [x, y] in metres, where x = column direction, y = row direction.
    ``ScanImage.global_center`` = physical_size / 2 = [W/2, H/2].

MATLAB engine (MapStruct, cell_corr_analysis):
    [row, col] in metres.
    ``vCenterG`` = ceil(N/2) * vPixSep per axis.
    ``vCenterL`` = same as vCenterG for full maps.
"""

from __future__ import annotations

import numpy as np

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration_matlab import (
    MapStruct,
    cell_corr_analysis,
)
from conversion.surface_comparison.models import (
    Cell,
    CellMetaData,
    ComparisonParams,
)
from conversion.surface_comparison.grid import _find_grid_origin


def _surface_map_to_map_struct(surface_map: ScanImage, angle: float = 0.0) -> MapStruct:
    """
    Convert a SurfaceMap to a MapStruct.

    MATLAB convention: vCenterG and vCenterL are [row, col] in metres.
    For a full map with no rotation:
        vCenterL = vCenterG = [ceil(nrows/2) * row_sep, ceil(ncols/2) * col_sep]

    :param surface_map: Application surface map.
    :param angle: Rotation angle in radians.
    :returns: MapStruct for the MATLAB engine.
    """
    nrows, ncols = surface_map.data.shape
    # MATLAB: vPixSep = [row_sep, col_sep] = [scale_y, scale_x]
    vPixSep = np.array([surface_map.scale_y, surface_map.scale_x])

    # MATLAB: vCenterL = ceil(N/2) * vPixSep
    vCenterL = np.array(
        [
            np.ceil(nrows / 2) * vPixSep[0],
            np.ceil(ncols / 2) * vPixSep[1],
        ]
    )

    # For a full, unregistered map, vCenterG = vCenterL
    vCenterG = vCenterL.copy()

    return MapStruct(
        map=surface_map.data.copy(),
        vCenterG=vCenterG,
        vCenterL=vCenterL,
        angle=angle,
        vPixSep=vPixSep,
    )


def _rc_to_xy(rc: np.ndarray) -> np.ndarray:
    """Convert [row, col] → [x, y]."""
    return np.array([rc[1], rc[0]])


def _xy_to_rc(xy: np.ndarray) -> np.ndarray:
    """Convert [x, y] → [row, col]."""
    return np.array([xy[1], xy[0]])


def register_cells(
    reference_map: ScanImage,
    comparison_map: ScanImage,
    params: ComparisonParams,
) -> list[Cell]:
    """
    Divide the reference into cells and register each against the comparison.

    This replaces the previous three-stage registration pipeline with the
    MATLAB-faithful ``cell_corr_analysis`` implementation, which performs:

    1. Coarse angular sweep — evaluates ACCF at discrete angles.
    2. Gradient-based ECC fine registration — refines [dx, dy, θ] iteratively.

    The grid origin is determined by ``_find_grid_origin``, which maximises
    valid-data coverage across all cells (faithfully translating MATLAB's
    ``cell_position_optim.m``). Its result is an [x, y] first-cell centre that
    is converted to [row, col] and passed to the MATLAB engine as
    ``vCellPosition``.

    :param reference_map: Fixed reference surface.
    :param comparison_map: Moving comparison surface.
    :param params: CMC algorithm parameters.
    :returns: List of per-cell results as ``Cell`` objects.
    """
    # Convert to MATLAB engine types
    map1 = _surface_map_to_map_struct(reference_map, angle=0.0)
    map2 = _surface_map_to_map_struct(comparison_map, angle=0.0)

    # Cell size: params uses (width, height) = (x, y), engine uses [row, col]
    vCellSize = np.array([params.cell_size[1], params.cell_size[0]])

    # --- Grid origin via coverage-maximising optimisation ---
    # _find_grid_origin returns the first-cell centre [x, y] in metres.
    # Convert to [row, col] for the MATLAB engine.
    origin_xy = _find_grid_origin(reference_map, params)
    vCellPosition = _xy_to_rc(origin_xy)

    # Angular search range: params uses degrees, engine uses radians
    shiftAngleMin = np.radians(params.search_angle_min)
    shiftAngleMax = np.radians(params.search_angle_max)

    # Run the MATLAB-faithful cell correlation analysis
    results = cell_corr_analysis(
        map1=map1,
        map2=map2,
        vCellSize=vCellSize,
        vCellPosition=vCellPosition,
        shiftAngleMin=shiftAngleMin,
        shiftAngleMax=shiftAngleMax,
        cellFillRefMin=params.minimum_fill_fraction,
        cellFillRegMin=max(0.1, params.minimum_fill_fraction - 0.15),
        cellFillRedMax=0.50,
        viParLevel=np.array([1]),
        viParLevelReg=np.array([1]),
        scale_min=1.0,
        scale_max=1.0,
        cellConvAnglePix=0.5,
        nCellRegImageReductionMax=4,
        nInterval=3,
        bEval180=False,
        verbose=False,
    )

    # Default CellMetaData before classification has run
    _default_meta = CellMetaData(
        is_outlier=False,
        residual_angle_deg=0.0,
        position_error=(0.0, 0.0),
    )

    # Convert engine results → Cell objects
    cell_results: list[Cell] = []
    for r in results:
        center_ref_xy = _rc_to_xy(r.vPos1)
        center_ref: tuple[float, float] = (
            float(center_ref_xy[0]),
            float(center_ref_xy[1]),
        )

        if r.bValid:
            center_comp_xy = _rc_to_xy(r.vPos2)
            center_comp: tuple[float, float] = (
                float(center_comp_xy[0]),
                float(center_comp_xy[1]),
            )
            angle_deg = float(np.degrees(r.dAngle))
            accf = float(r.accf)
        else:
            # TODO bValid should never be False. See cell_registration_matlab line 796 and below.
            center_comp = (float("nan"), float("nan"))
            angle_deg = float("nan")
            accf = float("nan")

        cell_results.append(
            Cell(
                center_reference=center_ref,
                center_comparison=center_comp,
                cell_data=r.cell_data if hasattr(r, "cell_data") else np.zeros((2, 2)),
                fill_fraction_reference=float(r.fill1),
                best_score=accf,
                angle_deg=angle_deg,
                is_congruent=False,
                meta_data=_default_meta.model_copy(),
            )
        )
        print(f"center ref {center_ref}")

    return cell_results
