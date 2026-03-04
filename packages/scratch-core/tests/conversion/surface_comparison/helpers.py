"""
Test helpers for cmc_classification tests.
"""

import numpy as np

from conversion.surface_comparison.models import (
    Cell,
    ComparisonParams,
    CellMetaData,
)


def as_array(value: int | float | list, dtype=np.float64) -> np.ndarray:
    """Convert a JSON value (which may contain ``None`` for NaN) to a numpy array."""
    if isinstance(value, (int, float)):
        return np.array([value], dtype=dtype)

    def _replace_none(v):
        return np.nan if v is None else v

    if isinstance(value, list) and value and isinstance(value[0], list):
        return np.array([[_replace_none(x) for x in row] for row in value], dtype=dtype)
    return np.array([_replace_none(x) for x in value], dtype=dtype)


def build_cells(inputs: dict) -> list[Cell]:
    """
    Build a list of Cell objects from a single MATLAB test-case input dict.

    MATLAB field mapping
    --------------------
    ============== ====================================================
    MATLAB field   Python Cell field
    ============== ====================================================
    vPos1          center_reference
    vPos2          center_comparison
    angle2-angle1  angle_reference  (delta; angle1 is always 0 in
                   our test data so angle_reference == angle2)
    simVal         best_score
    ============== ====================================================
    """
    mPos1 = as_array(inputs["mPos1"])
    mPos2 = as_array(inputs["mPos2"])
    angle1 = as_array(inputs["angle1"])
    angle2 = as_array(inputs["angle2"])
    sim_vals = as_array(inputs["simVals"])

    # Handle single-cell case where JSON gives 1-D arrays
    if mPos1.ndim == 1:
        mPos1 = mPos1.reshape(1, -1)
        mPos2 = mPos2.reshape(1, -1)
    if angle1.ndim == 0:
        angle1 = angle1.reshape(1)
        angle2 = angle2.reshape(1)
        sim_vals = sim_vals.reshape(1)

    n_cells = mPos1.shape[0]

    cells = []
    for i in range(n_cells):
        angle_val = float(angle2[i] - angle1[i])
        score_val = float(sim_vals[i])
        cell = Cell(
            cell_data=np.array([[0.0, 0.0], [0.1, 0.1]]),
            center_reference=mPos1[i],
            center_comparison=mPos2[i],
            angle_deg=angle_val,
            best_score=score_val,
            fill_fraction_reference=1.0,
            is_congruent=False,
            meta_data=CellMetaData(
                is_outlier=False,
                position_error=np.array([0.0, 0.0]),
                residual_angle_deg=0.0,
            ),
        )
        cells.append(cell)

    return cells


def build_test_params(
    inputs: dict,
) -> tuple[list[Cell], ComparisonParams, np.ndarray]:
    """
    Build a list of Cell instances, ComparisonParams, and global center from a MATLAB test-case input dict.

    Returns ``(cells, params, global_center)``.
    """
    cells = build_cells(inputs)

    params = ComparisonParams(
        correlation_threshold=inputs["simMin"],
        angle_threshold=inputs["angleMax"],
        position_threshold=inputs["distMax"],
    )

    global_center = np.array(inputs["vCenter"], dtype=np.float64)

    return cells, params, global_center
