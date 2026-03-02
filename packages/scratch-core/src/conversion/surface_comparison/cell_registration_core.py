"""
Cell registration adapter.

Bridges the application's ``SurfaceMap`` / ``ComparisonParams`` /
``CellResult`` types to the clean library-based registration engine
in ``cell_registration_clean``.

Coordinate conventions
----------------------
Application (SurfaceMap, CellResult):
    ``[x, y]`` in metres, where x = column direction, y = row direction.

Clean engine (cell_registration_clean):
    ``[row, col]`` in pixels internally, converted to metres at output.
    ``center_reference`` / ``center_comparison`` returned as ``[row, col]``
    in metres.
"""

from __future__ import annotations

import numpy as np

from conversion.surface_comparison.cell_registration import (
    CellRegistrationResult,
    _engine_register_cells,
)
from conversion.surface_comparison.models import (
    SurfaceMap,
    CellResult,
    ComparisonParams,
)


def _rc_to_xy(rc: np.ndarray) -> np.ndarray:
    """Convert [row, col] → [x, y]."""
    return np.array([rc[1], rc[0]])


def register_cells(
    reference_map: SurfaceMap,
    comparison_map: SurfaceMap,
    params: ComparisonParams,
    reference_map_processed: SurfaceMap | None = None,
    comparison_map_processed: SurfaceMap | None = None,
) -> list[CellResult]:
    """
    Divide the reference into cells and register each against the comparison.

    Uses the clean library-based pipeline:

    1. Coarse angular sweep — ``scipy.ndimage.rotate`` + ``match_template``
    2. Sub-pixel translation — ``phase_cross_correlation``
    3. ECC gradient refinement — ``cv2.findTransformECC``
    4. Final ACCF on processed data (if provided)

    :param reference_map: Fixed reference surface (leveled).
    :param comparison_map: Moving comparison surface (leveled).
    :param params: CMC algorithm parameters.
    :param reference_map_processed: Filtered reference for final ACCF
        (``commonEval='final'``). Falls back to ``reference_map`` if not given.
    :param comparison_map_processed: Filtered comparison for final ACCF.
        Falls back to ``comparison_map`` if not given.
    :returns: List of per-cell results.
    """
    # Use leveled data for registration, processed for final ACCF
    ref_lev = reference_map.data
    comp_lev = comparison_map.data

    ref_proc = (
        reference_map_processed.data if reference_map_processed is not None else ref_lev
    )
    comp_proc = (
        comparison_map_processed.data
        if comparison_map_processed is not None
        else comp_lev
    )

    # pixel_spacing: [row_spacing, col_spacing] = [scale_y, scale_x]
    pixel_spacing = np.array([reference_map.scale_y, reference_map.scale_x])

    # Cell size: params uses [width, height] = [x, y], engine uses [row, col]
    cell_size_m = np.array([params.cell_size[1], params.cell_size[0]])

    # Angular search range: params provides degrees directly
    angle_min = params.search_angle_min
    angle_max = params.search_angle_max
    angle_step = getattr(params, "search_angle_step", 1.0)

    # Run the clean registration engine
    engine_results: list[CellRegistrationResult] = _engine_register_cells(
        ref_leveled=ref_lev,
        comp_leveled=comp_lev,
        ref_processed=ref_proc,
        comp_processed=comp_proc,
        pixel_spacing=pixel_spacing,
        cell_size_m=cell_size_m,
        grid_origin_px=(0, 0),
        angle_min_deg=angle_min,
        angle_max_deg=angle_max,
        angle_step_deg=angle_step,
        min_fill=params.minimum_fill_fraction,
    )

    # Convert engine results → application CellResult objects
    cell_results = []
    for r in engine_results:
        # Engine returns [row, col] in metres → convert to [x, y]
        center_ref = _rc_to_xy(r.center_reference).astype(np.float64)

        if r.is_valid:
            center_comp = _rc_to_xy(r.center_comparison).astype(np.float64)
            angle = float(r.registration_angle)
            accf = float(r.accf)
        else:
            center_comp = np.array([np.nan, np.nan], dtype=np.float64)
            angle = float("nan")
            accf = float("nan")

        cell_results.append(
            CellResult(
                center_reference=center_ref,
                center_comparison=center_comp,
                registration_angle=angle,
                area_cross_correlation_function_score=accf,
                reference_fill_fraction=float(r.fill_fraction),
            )
        )

    return cell_results
