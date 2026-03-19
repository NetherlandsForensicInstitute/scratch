import numpy as np
import pytest
from skimage.transform import rotate
from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.coarse import (
    match_cells,
    _get_fill_fraction_map,
    _get_score_map,
    _unrotate_point,
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
        grid_cell = make_grid_cell(data=cell_data)

        # Act
        score_map = _get_score_map(
            comparison_array=filled,
            template=grid_cell.cell_data_filled,
            template_mask=grid_cell.valid_mask,
        )

        # Asserts
        peak_row, peak_col = np.unravel_index(np.argmax(score_map), score_map.shape)
        assert peak_row == CELL_TOP_LEFT[1]
        assert peak_col == CELL_TOP_LEFT[0]
        assert score_map[peak_row, peak_col] == pytest.approx(1.0)
        assert score_map.shape[0] == data.shape[0] - grid_cell.height + 1
        assert score_map.shape[1] == data.shape[1] - grid_cell.width + 1


class TestUnrotatePoint:
    # A non-square padded image to catch axis-swap bugs
    PADDED_WIDTH = 200
    PADDED_HEIGHT = 160
    POINT_IN_PADDED = (70.0, 40.0)

    def _padded_center(self) -> tuple[float, float]:
        return (self.PADDED_WIDTH - 1) / 2, (self.PADDED_HEIGHT - 1) / 2

    def _rotated_center(self, angle: float) -> tuple[float, float]:
        padded = np.zeros((self.PADDED_HEIGHT, self.PADDED_WIDTH))
        rotated = rotate(padded, angle=-angle, cval=0, order=0, resize=True)
        rot_h, rot_w = rotated.shape  # type: ignore
        return (rot_w - 1) / 2, (rot_h - 1) / 2

    def test_angle_zero_is_identity(self):
        # Arrange
        center = self._padded_center()

        # Act
        x, y = _unrotate_point(
            rotated_point=self.POINT_IN_PADDED,
            original_image_center=center,
            rotated_image_center=center,
            angle_deg=0.0,
        )

        # Assert
        assert x == pytest.approx(self.POINT_IN_PADDED[0])
        assert y == pytest.approx(self.POINT_IN_PADDED[1])

    @pytest.mark.parametrize("angle", [30.0, -60.0, 90.0])
    def test_round_trip_recovers_original_point(self, angle: float):
        """
        Analytically forward-map a known point through skimage.rotate(padded, -angle, resize=True),
        then verify _unrotate_point recovers it exactly.
        """
        # Arrange
        cx_pad, cy_pad = self._padded_center()
        cx_rot, cy_rot = self._rotated_center(angle)
        px, py = self.POINT_IN_PADDED
        a = np.radians(angle)
        fwd_x = np.cos(a) * (px - cx_pad) - np.sin(a) * (py - cy_pad) + cx_rot
        fwd_y = np.sin(a) * (px - cx_pad) + np.cos(a) * (py - cy_pad) + cy_rot

        # Act
        recovered_x, recovered_y = _unrotate_point(
            rotated_point=(fwd_x, fwd_y),
            original_image_center=(cx_pad, cy_pad),
            rotated_image_center=(cx_rot, cy_rot),
            angle_deg=angle,
        )

        # Assert
        assert recovered_x == pytest.approx(px, abs=1e-6)
        assert recovered_y == pytest.approx(py, abs=1e-6)


class TestNegativeCorrelation:
    def test_coarse_registration_can_find_negative_correlation(self):
        # Arrange
        # Use a non-periodic surface so the inverted cell has no spurious positive correlations elsewhere
        image_data = make_surface(height=30, width=40, scale=1.0)
        comparison_data = -image_data  # Exact pointwise inversion
        comparison_image = ScanImage(data=comparison_data, scale_x=1.0, scale_y=1.0)
        # Make the cell size large enough to not allow for spurious correlations
        cell_data = image_data[5:25, 5:35]
        grid_cell = make_grid_cell(data=cell_data)
        params = ComparisonParams(
            search_angle_min=0,
            search_angle_max=0,
            search_angle_step=1,
            minimum_fill_fraction=1,
        )
        # Act
        results = match_cells(
            grid_cells=[grid_cell], comparison_image=comparison_image, params=params
        )
        # Assert
        assert results[0].best_score < 0.0
