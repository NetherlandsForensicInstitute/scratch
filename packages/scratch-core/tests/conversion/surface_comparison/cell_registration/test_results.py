"""Tests for conversion.surface_comparison.cell_registration.results."""

import pytest

from conversion.surface_comparison.cell_registration.results import (
    convert_grid_cell_to_cell,
)
from conversion.surface_comparison.models import GridCell

PIXEL_SIZE = 1e-6


class TestConvertGridCellToCell:
    def test_convert_grid_cell_to_cell_centers_in_meters(
        self, fully_valid_grid_cell: GridCell
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert — reference center must equal (top_left + half_cell) * pixel_size
        expected_cx = (cell.top_left[0] + cell.width / 2) * PIXEL_SIZE
        expected_cy = (cell.top_left[1] + cell.height / 2) * PIXEL_SIZE
        assert result.center_reference == pytest.approx((expected_cx, expected_cy))

    def test_convert_grid_cell_to_cell_score_propagated(
        self, fully_valid_grid_cell: GridCell
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert
        assert result.best_score == pytest.approx(cell.grid_search_params.score)

    def test_convert_grid_cell_to_cell_fill_fraction_propagated(
        self, fully_valid_grid_cell: GridCell
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert
        assert result.fill_fraction_reference == pytest.approx(cell.fill_fraction)
