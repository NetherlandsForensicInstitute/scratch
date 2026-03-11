"""Tests for conversion.surface_comparison.cell_registration.match.

Each test uses an identical reference and comparison image so the expected
best score is ~1.0 and the best position coincides with the cell's own
top-left corner.  Two magnitude scales are exercised (unit-scale and µm-scale)
to confirm numerical robustness.
"""

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.match import (
    match_cells,
    _build_templates,
    _get_fill_fraction_map,
    _get_score_map,
)
from conversion.surface_comparison.models import GridCell, ComparisonParams

from .helpers import (
    make_scan_image,
    make_grid_cell,
    make_surface,
    identity_params,
)


IMAGE_HEIGHT = 80
IMAGE_WIDTH = 98
CELL_SIZE = 20
PIXEL_SIZE = 1e-6
CELL_TOP_LEFT = (40, 30)
SCORE_TOLERANCE = 0.05
FILL_FRACTION_THRESHOLD = 0.5


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
            comparison_image=comparison_image,
            params=params,
            fill_value_reference=float(np.nanmean(comparison_image.data)),
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
            comparison_image=comparison_image,
            params=params,
            fill_value_reference=float(np.nanmean(comparison_image.data)),
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
            comparison_image=comparison_image,
            params=params,
            fill_value_reference=float(np.nanmean(comparison_image.data)),
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
            comparison_image=comparison_image,
            params=params,
            fill_value_reference=float(np.nanmean(comparison_image.data)),
        )

        # Assert
        assert cells[0].center_comparison == cells[0].center_reference

    def test_match_cells_empty_input_returns_empty_list(self):
        # Arrange
        comparison_image = make_scan_image(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        params = identity_params(cell_size_px=CELL_SIZE)

        # Act
        cells = match_cells(
            grid_cells=[],
            comparison_image=comparison_image,
            params=params,
            fill_value_reference=0.0,
        )

        # Assert
        assert cells == []


class TestGetFillFraction:
    def test_get_fill_fraction_map_all_valid(self):
        # Arrange
        valid_mask = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)

        # Act
        fill_map = _get_fill_fraction_map(
            valid_pixel_mask=valid_mask,
            cell_height=CELL_SIZE,
            cell_width=CELL_SIZE,
        )

        # Assert — interior positions (fully covered window) should be 1.0
        interior = fill_map[
            : IMAGE_HEIGHT - CELL_SIZE + 1, : IMAGE_WIDTH - CELL_SIZE + 1
        ]
        np.testing.assert_allclose(interior, 1.0, atol=1e-6)


class TestGetScoreMap:
    def test_get_score_map_self_match_peak_at_cell_top_left(self):
        # Arrange
        data = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, scale=1.0)
        # Fill NaN so cv2 works on a clean float32 array
        mean_val = float(np.nanmean(data))
        filled = np.where(np.isnan(data), mean_val, data).astype(np.float32)
        cell_data = data[
            CELL_TOP_LEFT[1] : CELL_TOP_LEFT[1] + CELL_SIZE,
            CELL_TOP_LEFT[0] : CELL_TOP_LEFT[0] + CELL_SIZE,
        ]
        grid_cell = make_grid_cell(data=np.nan_to_num(cell_data, nan=mean_val))

        # Act
        score_map = _get_score_map(comparison_array_filled=filled, cell=grid_cell)

        # Assert
        peak_row, peak_col = np.unravel_index(np.argmax(score_map), score_map.shape)
        assert peak_row == CELL_TOP_LEFT[1]
        assert peak_col == CELL_TOP_LEFT[0]


class TestBuildTemplates:
    def test_build_templates_fills_nans(self):
        # Arrange
        data = make_surface(height=CELL_SIZE, width=CELL_SIZE)
        data[0, 0] = np.nan
        grid_cell = make_grid_cell(data=data)
        fill_value = 0.0

        # Act
        templates = _build_templates(grid_cells=[grid_cell], fill_value=fill_value)

        # Assert
        assert not np.any(np.isnan(templates[0].cell_data))

    def test_build_templates_does_not_mutate_original(self):
        # Arrange
        data = make_surface(height=CELL_SIZE, width=CELL_SIZE)
        data[0, 0] = np.nan
        grid_cell = make_grid_cell(data=data)

        # Act
        _build_templates(grid_cells=[grid_cell], fill_value=0.0)

        # Assert — original cell_data must still contain the NaN
        assert np.isnan(grid_cell.cell_data[0, 0])
