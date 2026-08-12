"""Tests for conversion.surface_comparison.cell_registration.stage_builders."""

import numpy as np
import pytest
from skimage.transform import rotate

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.stage_builders import (
    build_coarse_stage,
    convert_grid_cell_to_cell,
)
from conversion.surface_comparison.models import ComparisonParams, GridCell
from conversion.surface_comparison.pipeline import resample_to_scale

from .helpers import (
    identity_params,
    make_grid_cell,
    make_scan_image,
    register_cells,
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
        cells = register_cells(
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
        cells = register_cells(
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
        cells = register_cells(
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
        cells = register_cells(
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
        cells = register_cells(
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
            register_cells(
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
        cells = register_cells(
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
        results = register_cells(
            grid_cells=[grid_cell],
            reference_image=comparison_image,
            comparison_image=comparison_image,
            params=params,
        )

        # Assert
        assert results[0].best_score < 0.0


class TestBuildCoarseStage:
    def test_coarse_canvas_lands_on_the_reference_scale_regardless_of_comparison_source(self):
        """
        Regression test: the comparison canvas must land on the same physical coarse grid as
        the reference, whether the comparison image passed in is the original scan (its own
        native pixel scale) or one already resampled onto the reference's scale.

        Previously, downsampling the *original* comparison image by ``cap_factor`` alone (no
        adjustment for its native scale relative to the reference) produced a coarse canvas at
        the wrong physical pixel pitch whenever the two scans had different native resolutions.
        """
        # Arrange: reference and comparison cover the same physical scene, but the comparison
        # scan's native pixel pitch is half the reference's (twice the pixel density).
        rng = np.random.default_rng(0)
        reference_image = ScanImage(
            data=rng.random((200, 200)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        comparison_original = ScanImage(
            data=rng.random((400, 400)).astype(np.float64), scale_x=0.5, scale_y=0.5
        )
        comparison_aligned = resample_to_scale(
            comparison_original, reference_image.scale_x
        )
        grid_cell = make_grid_cell(
            data=reference_image.data[50:70, 50:70], top_left=(50, 50)
        )
        cap_factor = 5 / 3

        # Act
        from_original = build_coarse_stage(
            comparison_original, reference_image, [grid_cell], cap_factor
        )
        from_aligned = build_coarse_stage(
            comparison_aligned, reference_image, [grid_cell], cap_factor
        )

        # Assert
        assert from_original.image.shape == from_aligned.image.shape

    def test_coarse_canvas_matches_cap_factor_when_already_aligned(self):
        # Arrange: comparison already on the reference's scale, so scale_match_factor == 1 and
        # this reduces to a plain downsample by cap_factor, as before.
        rng = np.random.default_rng(1)
        reference_image = ScanImage(
            data=rng.random((120, 120)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        comparison_image = ScanImage(
            data=rng.random((120, 120)).astype(np.float64), scale_x=1.0, scale_y=1.0
        )
        grid_cell = make_grid_cell(
            data=reference_image.data[10:30, 10:30], top_left=(10, 10)
        )
        cap_factor = 3.0

        # Act
        coarse = build_coarse_stage(
            comparison_image, reference_image, [grid_cell], cap_factor
        )

        # Assert
        expected_side = max(1, int(np.ceil(120 / cap_factor)))
        coarse_cell = max(1, int(np.ceil(grid_cell.width / cap_factor)))
        assert coarse.image.shape == (
            expected_side + 2 * coarse_cell,
            expected_side + 2 * coarse_cell,
        )


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
