"""Tests for conversion.surface_comparison.cell_registration.core.

coarse_registration is tested end-to-end: a grid cell extracted from the
reference image is registered against an identical comparison image so the
expected best score is near 1.0 and the angle is 0°.

fine_registration is a stub that must return its input unchanged.
"""

import numpy as np
import pytest
from skimage.transform import rotate
from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.core import (
    coarse_registration,
)
from conversion.surface_comparison.models import GridCell, ComparisonParams
from tests.conversion.surface_comparison.cell_registration.helpers import (
    plot_cell_registration_results,
)

SCORE_TOLERANCE = 0.05


def test_coarse_registration_returns_one_cell_per_grid_cell(
    identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, comparison_image, params = identical_match_inputs

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert len(cells) == len(grid_cells)


def test_coarse_registration_self_match_score_near_one(
    identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, comparison_image, params = identical_match_inputs

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert cells[0].best_score >= 1.0 - SCORE_TOLERANCE


def test_coarse_registration_self_match_angle_is_zero(
    identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, comparison_image, params = identical_match_inputs

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert cells[0].angle_deg == pytest.approx(0.0)


@pytest.mark.parametrize("angle", [0, 60, -40])
def test_coarse_registration_self_match_angle_is_found(
    identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    angle: float,
    plot: bool = False,
):
    # Arrange
    angle_min, angle_max, angle_step = -80, 80, 20
    grid_cells, reference_image, _ = identical_match_inputs
    rotated = rotate(
        reference_image.data,
        angle=angle,
        order=0,
        resize=False,
        cval=np.nan,  # type: ignore
    )
    comparison_image = reference_image.model_copy(update={"data": rotated})

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=ComparisonParams(
            search_angle_min=angle_min,
            search_angle_max=angle_max,
            search_angle_step=angle_step,
        ),
    )

    if plot:
        plot_cell_registration_results(
            reference_image=reference_image,
            comparison_image=comparison_image,
            cells=cells,
        )

    # Assert
    assert cells[0].angle_deg == pytest.approx(angle)
