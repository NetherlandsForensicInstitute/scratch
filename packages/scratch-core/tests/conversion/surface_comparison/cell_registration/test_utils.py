"""Tests for conversion.surface_comparison.cell_registration.utils."""

import numpy as np
import pytest

from conversion.surface_comparison.cell_registration.utils import (
    compute_fill_fraction,
    convert_grid_cell_to_cell,
    pad_image_array,
)
from conversion.surface_comparison.models import GridCell

from .helpers import make_surface


CELL_SIZE = 50
PIXEL_SIZE = 1e-6
IMAGE_HEIGHT = 850
IMAGE_WIDTH = 600
PAD_HEIGHT = 25
PAD_WIDTH = 25


class TestPadImageArray:
    def test_pad_image_array_output_shape(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert
        assert padded.shape == (
            IMAGE_HEIGHT + 2 * PAD_HEIGHT,
            IMAGE_WIDTH + 2 * PAD_WIDTH,
        )

    def test_pad_image_array_border_is_nan(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert — all border rows/cols should be NaN (default fill)
        assert np.all(np.isnan(padded[:PAD_HEIGHT, :]))
        assert np.all(np.isnan(padded[-PAD_HEIGHT:, :]))
        assert np.all(np.isnan(padded[:, :PAD_WIDTH]))
        assert np.all(np.isnan(padded[:, -PAD_WIDTH:]))

    def test_pad_image_array_interior_matches_original(self):
        # Arrange
        array = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        padded = pad_image_array(array, pad_width=PAD_WIDTH, pad_height=PAD_HEIGHT)

        # Assert
        interior = padded[
            PAD_HEIGHT : PAD_HEIGHT + IMAGE_HEIGHT, PAD_WIDTH : PAD_WIDTH + IMAGE_WIDTH
        ]
        np.testing.assert_array_equal(interior, array)


class TestComputeFillFraction:
    def test_compute_fill_fraction_fully_valid(self):
        # Arrange
        array = make_surface(height=CELL_SIZE, width=CELL_SIZE)

        # Act
        fraction = compute_fill_fraction(array)

        # Assert
        assert fraction == pytest.approx(1.0)

    def test_compute_fill_fraction_half_nan(self):
        # Arrange
        array = make_surface(height=CELL_SIZE, width=CELL_SIZE)
        array[: CELL_SIZE // 2, :] = np.nan

        # Act
        fraction = compute_fill_fraction(array)

        # Assert
        assert fraction == pytest.approx(0.5)


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
        self,
        fully_valid_grid_cell: GridCell,
    ):
        # Arrange
        cell = fully_valid_grid_cell

        # Act
        result = convert_grid_cell_to_cell(grid_cell=cell, pixel_size=PIXEL_SIZE)

        # Assert
        assert result.fill_fraction_reference == pytest.approx(cell.fill_fraction)
