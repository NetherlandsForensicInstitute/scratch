"""Tests for conversion.surface_comparison.cell_registration.match_matlab.

The self-matching tests verify that an identical reference and comparison image
yield a score near 1.0 at the correct position.  Both unit-scale and µm-scale
inputs are exercised so numerical robustness across orders of magnitude is
confirmed.  A dedicated test for _nan_aware_ncc_map ensures NaN regions in
the template do not produce spurious scores.
"""

import numpy as np
import pytest

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.match_matlab import (
    match_cells,
    _nan_aware_ncc_map,
)
from conversion.surface_comparison.models import GridCell, ComparisonParams

from .helpers import (
    make_scan_image,
    make_surface,
    identity_params,
)


IMAGE_HEIGHT = 950
IMAGE_WIDTH = 800
CELL_SIZE = 80
PIXEL_SIZE = 1e-6
CELL_TOP_LEFT = (30, 30)
SCORE_TOLERANCE = 0.05
NAN_FILL_ROWS = 4  # number of rows set to NaN in partial-NaN tests
MIN_OVERLAP = 10


class TestMatch:
    def test_match_cells_returns_one_cell_per_grid_cell(
        self,
        identical_match_inputs: tuple[list[GridCell], ScanImage, ComparisonParams],
    ):
        # Arrange
        grid_cells, comparison_image, params = identical_match_inputs

        # Act
        cells = match_cells(
            grid_cells=grid_cells, comparison_image=comparison_image, params=params
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
            grid_cells=grid_cells, comparison_image=comparison_image, params=params
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
            grid_cells=grid_cells, comparison_image=comparison_image, params=params
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
            grid_cells=[], comparison_image=comparison_image, params=params
        )

        # Assert
        assert cells == []


class TestNaNAwareCorrelation:
    def test_nan_aware_ncc_map_self_match_score_is_one(self):
        # Arrange
        template = make_surface(height=CELL_SIZE, width=CELL_SIZE, scale=1e-6)
        padded_image = np.pad(
            template, pad_width=CELL_SIZE // 2, constant_values=np.nan
        )

        # Act
        score_map = _nan_aware_ncc_map(
            image=padded_image,
            template=template,
            min_overlap=MIN_OVERLAP,
        )

        # Assert — peak must be exactly 1.0 at the centred position
        assert np.max(score_map) == pytest.approx(1.0, abs=1e-5)

    def test_nan_aware_ncc_map_output_contains_no_nans(self):
        # Arrange
        template = make_surface(height=CELL_SIZE, width=CELL_SIZE)
        template[:NAN_FILL_ROWS, :] = np.nan  # partial NaN in template
        image = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        score_map = _nan_aware_ncc_map(
            image=image, template=template, min_overlap=MIN_OVERLAP
        )

        # Assert
        assert not np.any(np.isnan(score_map))

    def test_nan_aware_ncc_map_scores_clipped_to_unit_range(self):
        # Arrange
        template = make_surface(height=CELL_SIZE, width=CELL_SIZE, scale=1e-6)
        image = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, scale=1e-6)

        # Act
        score_map = _nan_aware_ncc_map(
            image=image, template=template, min_overlap=MIN_OVERLAP
        )

        # Assert
        assert np.all(score_map >= -1.0)
        assert np.all(score_map <= 1.0)

    def test_nan_aware_ncc_map_low_overlap_positions_score_zero(self):
        # Arrange — template with only the bottom-right corner valid (tiny overlap everywhere)
        template = np.full((CELL_SIZE, CELL_SIZE), np.nan)
        template[-2:, -2:] = make_surface(height=2, width=2)
        image = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        # Require more overlap than any position can provide
        high_min_overlap = 100

        # Act
        score_map = _nan_aware_ncc_map(
            image=image, template=template, min_overlap=high_min_overlap
        )

        # Assert — all positions gated → all scores are 0
        np.testing.assert_array_equal(score_map, 0.0)

    def test_nan_aware_ncc_map_output_shape(self):
        # Arrange
        template = make_surface(height=CELL_SIZE, width=CELL_SIZE)
        image = make_surface(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

        # Act
        score_map = _nan_aware_ncc_map(
            image=image, template=template, min_overlap=MIN_OVERLAP
        )

        # Assert
        expected_shape = (IMAGE_HEIGHT - CELL_SIZE + 1, IMAGE_WIDTH - CELL_SIZE + 1)
        assert score_map.shape == expected_shape
