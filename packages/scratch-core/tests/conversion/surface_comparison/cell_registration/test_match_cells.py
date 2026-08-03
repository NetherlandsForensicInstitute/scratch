import numpy as np
import pytest
from skimage.transform import rotate

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.match_cells import match_cells
from conversion.surface_comparison.models import GridCell, ComparisonParams
from .helpers import (
    make_scan_image,
    identity_params,
    make_grid_cell,
)

SCORE_TOLERANCE = 0.05
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
            grid_cells=grid_cells, comparison_image=comparison_image, params=params
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

    @pytest.mark.parametrize("reduction", [None, 4], ids=["exhaustive", "coarse"])
    @pytest.mark.parametrize("angle", [0, 60, -40])
    def test_match_cells_recovers_a_large_rotation(
        self, identical_match_inputs, angle, reduction
    ):
        # Arrange: a sweep far wider than production uses, since marks are occasionally
        # presented at a wholly different orientation.
        grid_cells, reference_image, _ = identical_match_inputs
        rotated = rotate(
            reference_image.data,
            angle=angle,
            order=0,
            resize=True,
            cval=np.nan,  # type: ignore[arg-type]  # skimage stub types cval as int
        )
        comparison_image = reference_image.model_copy(update={"data": rotated})

        # Act
        cells = match_cells(
            grid_cells=grid_cells,
            comparison_image=comparison_image,
            params=ComparisonParams(
                search_angle_min=-80, search_angle_max=80, search_angle_step=20
            ),
            reduction=reduction,
        )

        # Assert
        assert cells[0].angle_deg == pytest.approx(angle)


class TestNegativeCorrelation:
    @pytest.mark.parametrize("reduction", [None, 4], ids=["exhaustive", "coarse"])
    def test_registration_can_find_negative_correlation(self, reduction):
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
        )

        # Act
        results = match_cells(
            grid_cells=[grid_cell],
            comparison_image=comparison_image,
            params=params,
            reduction=reduction,
        )

        # Assert
        assert results[0].best_score < 0.0
