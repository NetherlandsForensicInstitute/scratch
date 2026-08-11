"""Tests for conversion.surface_comparison.cell_registration.match_cells."""

import numpy as np
import pytest
from skimage.transform import rotate

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.match_cells import (
    convert_grid_cell_to_cell,
    match_cells,
)
from conversion.surface_comparison.models import ComparisonParams, GridCell

from .helpers import (
    identity_params,
    make_grid_cell,
    make_scan_image,
)

SCORE_TOLERANCE = 0.05
PIXEL_SIZE = 1e-6
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 98
CELL_SIZE = 20


class TestMatch:
    def test_match_cells_returns_one_cell_per_grid_cell(
        self,
        identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    ):
        # Arrange
        grid_cells, comparison_image, params = identical_match_inputs

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert len(cells) == len(grid_cells)

    def test_match_cells_self_match_score_near_one(
        self,
        identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    ):
        # Arrange
        grid_cells, comparison_image, params = identical_match_inputs

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert cells[0].best_score >= 1.0 - SCORE_TOLERANCE

    def test_match_cells_self_match_angle_is_zero(
        self,
        identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    ):
        # Arrange
        grid_cells, comparison_image, params = identical_match_inputs

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert cells[0].angle_deg == pytest.approx(0.0)

    def test_match_cells_self_match_center_is_equal(
        self,
        identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    ):
        # Arrange
        grid_cells, comparison_image, params = identical_match_inputs

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert cells[0].center_comparison == cells[0].center_reference

    def test_match_cells_empty_input_returns_empty_list(self):
        # Arrange
        comparison_image = make_scan_image(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        cells = match_cells(
            grid_cells=[],
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=identity_params(),
        )

        # Assert
        assert cells == []

    def test_match_cells_rejects_a_pixel_scale_mismatch(self, identical_match_inputs):
        # Arrange: the pipeline is responsible for bringing both images onto one grid.
        grid_cells, reference_image, params = identical_match_inputs
        comparison_image = reference_image.model_copy(
            update={"scale_x": reference_image.scale_x * 2}
        )

        # Act / Assert
        with pytest.raises(ValueError, match="same pixel scale"):
            match_cells(
                grid_cells=grid_cells,
                reference_image=reference_image,
                comparison_image=comparison_image,
                params=params,
            )

    @pytest.mark.parametrize("max_size", [1000, 800], ids=["exhaustive", "coarse"])
    @pytest.mark.parametrize("angle", [0, 60, -40])
    def test_match_cells_recovers_a_large_rotation(
        self, identical_match_inputs, angle, max_size
    ):
        # Arrange: a sweep far wider than production uses, since marks are occasionally
        # presented at a wholly different orientation.
        grid_cells, reference_image, _ = identical_match_inputs

        # order=1 (bilinear) prevents pixelation artifacts during rotation;
        # resize=True expands canvas to preserve all grid cell features without clipping.
        rotated = rotate(
            reference_image.data,
            angle=angle,
            order=1,
            resize=True,
            cval=np.nan,  # type: ignore[arg-type]
        )
        comparison_image = reference_image.model_copy(update={"data": rotated})

        params = ComparisonParams(
            search_angle_min=-80,
            search_angle_max=80,
            search_angle_step=20,
            max_size=max_size,
        )

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            reference_image=reference_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert cells[0].angle_deg == pytest.approx(angle)


class TestNegativeCorrelation:
    @pytest.mark.parametrize("max_size", [1000, 20], ids=["exhaustive", "coarse"])
    def test_registration_can_find_negative_correlation(self, max_size):
        # Arrange: a non-periodic surface, so the inverted cell has no spurious positive
        # correlations elsewhere. Negating the cell rather than the comparison keeps the helper
        # usable and gives a bit-identical score map, since Pearson correlation flips sign
        # whichever operand is inverted.
        comparison_image = make_scan_image(
            height=30, width=40, scale=1.0, pixel_size=1.0
        )
        # Large enough that no spurious match can outscore the true inversion.
        cell_data = -comparison_image.data[2:28, 2:38]
        grid_cell = make_grid_cell(data=cell_data)

        params = ComparisonParams(
            search_angle_min=0,
            search_angle_max=0,
            search_angle_step=1,
            minimum_fill_fraction=1,
            max_size=max_size,
        )

        # Act
        results = match_cells(
            grid_cells=[grid_cell],
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert results[0].best_score < 0.0


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
