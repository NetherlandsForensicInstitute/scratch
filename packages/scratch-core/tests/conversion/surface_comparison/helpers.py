"""
Test helpers for cmc_classification tests.
"""

import numpy as np

from conversion.surface_comparison.models import Cell, CellMetaData, ComparisonParams

# Placeholder cell data: a minimal 2x2 height map, unused by the classifier.
_PLACEHOLDER_CELL_DATA = np.array([[0.0, 0.0], [0.1, 0.1]])

# Default CellMetaData used before the classifier has run.
_DEFAULT_META_DATA = CellMetaData(
    is_outlier=False,
    residual_angle_deg=0.0,
    position_error=(0.0, 0.0),
)


def build_cells(inputs: dict) -> list[Cell]:
    """
    Build a list of Cell objects from a test-case input dict.

    The angle stored on each Cell is the delta ``angles_comparison - angles_reference``.
    In all current test cases ``angles_reference`` is zero, so the delta equals
    ``angles_comparison`` directly.
    """
    centers_reference = np.array(inputs["centers_reference"])
    centers_comparison = np.array(inputs["centers_comparison"])
    angles_reference = np.atleast_1d(np.array(inputs["angles_reference"], dtype=float))
    angles_comparison = np.atleast_1d(
        np.array(inputs["angles_comparison"], dtype=float)
    )
    correlation_scores = np.atleast_1d(
        np.array(inputs["correlation_scores"], dtype=float)
    )

    if centers_reference.ndim == 1:
        centers_reference = centers_reference.reshape(1, -1)
        centers_comparison = centers_comparison.reshape(1, -1)

    return [
        Cell(
            cell_data=_PLACEHOLDER_CELL_DATA,
            center_reference=(
                float(centers_reference[i, 0]),
                float(centers_reference[i, 1]),
            ),
            center_comparison=(
                float(centers_comparison[i, 0]),
                float(centers_comparison[i, 1]),
            ),
            angle_deg=float(angles_comparison[i] - angles_reference[i]),
            best_score=float(correlation_scores[i]),
            fill_fraction_reference=1.0,
            is_congruent=False,
            meta_data=_DEFAULT_META_DATA.model_copy(),
        )
        for i in range(centers_reference.shape[0])
    ]


def build_test_inputs(
    inputs: dict,
) -> tuple[list[Cell], ComparisonParams, tuple[float, float]]:
    """
    Build the full set of inputs for ``classify_congruent_cells`` from a test-case
    input dict.

    Returns ``(cells, params, rotation_center)``.
    """
    cells = build_cells(inputs)

    params = ComparisonParams(
        correlation_threshold=inputs["correlation_threshold"],
        angle_threshold=inputs["angle_threshold"],
        position_threshold=inputs["position_threshold"],
    )

    rotation_center_list = inputs["rotation_center"]
    rotation_center = (float(rotation_center_list[0]), float(rotation_center_list[1]))

    return cells, params, rotation_center


def _make_cell(
    angle_deg: float,
    center_reference: tuple[float, float] = (0.0, 0.0),
    center_comparison: tuple[float, float] = (0.0, 0.0),
    best_score: float = 0.8,
    is_outlier: bool = False,
    residual_angle_deg: float = 0.0,
    position_error: tuple[float, float] = (0.0, 0.0),
) -> Cell:
    """Construct a minimal Cell with sensible defaults for unit testing."""
    return Cell(
        center_reference=center_reference,
        cell_data=np.zeros((4, 4)),
        fill_fraction_reference=1.0,
        best_score=best_score,
        angle_deg=angle_deg,
        center_comparison=center_comparison,
        is_congruent=False,
        meta_data=CellMetaData(
            is_outlier=is_outlier,
            residual_angle_deg=residual_angle_deg,
            position_error=position_error,
        ),
    )
