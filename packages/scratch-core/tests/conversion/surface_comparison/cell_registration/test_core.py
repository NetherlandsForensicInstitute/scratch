"""Tests for conversion.surface_comparison.cell_registration.core.

coarse_registration is tested end-to-end: a grid cell extracted from the
reference image is registered against an identical comparison image so the
expected best score is near 1.0 and the angle is 0°.

fine_registration is a stub that must return its input unchanged.
"""

import pytest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.core import (
    coarse_registration,
)
from conversion.surface_comparison.models import GridCell, ComparisonParams

SCORE_TOLERANCE = 0.05


def test_coarse_registration_returns_one_cell_per_grid_cell(
    identical_registration_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, reference_image, params = identical_registration_inputs
    comparison_image = reference_image  # identical images

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert len(cells) == len(grid_cells)


def test_coarse_registration_self_match_score_near_one(
    identical_registration_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, reference_image, params = identical_registration_inputs
    comparison_image = reference_image

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert cells[0].best_score >= 1.0 - SCORE_TOLERANCE


def test_coarse_registration_self_match_angle_is_zero(
    identical_registration_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
):
    # Arrange
    grid_cells, reference_image, params = identical_registration_inputs
    comparison_image = reference_image

    # Act
    cells = coarse_registration(
        grid_cells=grid_cells,
        comparison_image=comparison_image,
        params=params,
    )

    # Assert
    assert cells[0].angle_deg == pytest.approx(0.0)
