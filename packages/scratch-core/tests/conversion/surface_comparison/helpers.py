"""
Test helpers for cmc_classification tests.
"""

import numpy as np

from conversion.surface_comparison.models import Cell, CellMetaData, ComparisonParams

# Placeholder cell data: a minimal 2x2 height map, unused by the classifier.
_PLACEHOLDER_CELL_SIZE = (1e-3, 1e-3)

# Default CellMetaData used before the classifier has run.
_DEFAULT_META_DATA = CellMetaData(
    is_outlier=False,
    residual_angle_deg=0.0,
    position_error=(0.0, 0.0),
)


def _reflect_y_axis(coordinates: np.ndarray) -> np.ndarray:
    """
    Reflect y-axis by multiplying data with reflection matrix.
    """

    reflection_matrix = np.array([[1, 0], [0, -1]]).transpose()

    return coordinates @ reflection_matrix


def build_cells(inputs: dict) -> list[Cell]:
    """
    Build a list of Cell objects from a test-case input dict.

    The angle stored on each Cell is the delta ``angles_comparison - angles_reference``.
    In all current test cases ``angles_reference`` is zero, so the delta equals
    ``angles_comparison`` directly.
    Reflect y-coordinates since matlab uses mathematical coordinates and our pipeline uses image_coordinates
    """
    centers_reference = np.array(inputs["centers_reference"])
    centers_comparison = np.array(inputs["centers_comparison"])

    # also works when ndim = 1
    centers_reference = _reflect_y_axis(centers_reference)
    centers_comparison = _reflect_y_axis(centers_comparison)

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
            cell_size=_PLACEHOLDER_CELL_SIZE,
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
    Reflect y-coordinate of rotation_center since matlab uses 'mathematical coordinates and our pipeline uses image_coordinates
    Returns ``(cells, params, rotation_center)``.
    """
    cells = build_cells(inputs)

    params = ComparisonParams(
        correlation_threshold=inputs["correlation_threshold"],
        angle_deviation_threshold=inputs["angle_deviation_threshold"],
        position_threshold=inputs["position_threshold"],
    )

    rotation_center_list = inputs["rotation_center"]
    rotation_center = (float(rotation_center_list[0]), -float(rotation_center_list[1]))

    return cells, params, rotation_center
