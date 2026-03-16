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
            comparison_array=filled, template=grid_cell.cell_data_filled
        )

        # Asserts
        peak_row, peak_col = np.unravel_index(np.argmax(score_map), score_map.shape)
        assert peak_row == CELL_TOP_LEFT[1]
        assert peak_col == CELL_TOP_LEFT[0]
        assert score_map[peak_row, peak_col] == pytest.approx(1.0)
        assert score_map.shape[0] == data.shape[0] - grid_cell.height + 1
        assert score_map.shape[1] == data.shape[1] - grid_cell.width + 1


class TestUnrotatePoint:
    PADDED_WIDTH = 200
    PADDED_HEIGHT = 160
    ROTATED_WIDTH = 200  # same as padded at angle=0
    ROTATED_HEIGHT = 160
    POINT_IN_PADDED = (70.0, 40.0)  # a known point in the padded image

    def test_angle_zero_is_identity(self):
        # Arrange — at angle=0 the rotated image has the same shape as the input,
        # so unrotate should return the input point unchanged.
        rotated_point = self.POINT_IN_PADDED

        # Act
        x, y = _unrotate_point(
            rotated_point=rotated_point,
            original_image_size=(self.PADDED_WIDTH, self.PADDED_HEIGHT),
            rotated_image_size=(self.ROTATED_WIDTH, self.ROTATED_HEIGHT),
            angle_deg=0.0,
        )

        # Assert
        assert x == pytest.approx(rotated_point[0])
        assert y == pytest.approx(rotated_point[1])

    @pytest.mark.parametrize("angle", [30.0, 60.0, 90.0])
    def test_round_trip_recovers_original_point(self, angle: float):
        # Arrange
        padded = np.zeros((self.PADDED_HEIGHT, self.PADDED_WIDTH))
        rotated = rotate(padded, angle=-angle, cval=0, order=0, resize=True)
        rot_h, rot_w = rotated.shape  # type: ignore

        cx_pad = (self.PADDED_WIDTH - 1) / 2
        cy_pad = (self.PADDED_HEIGHT - 1) / 2
        cx_rot = (rot_w - 1) / 2
        cy_rot = (rot_h - 1) / 2

        px, py = self.POINT_IN_PADDED
        a = np.radians(angle)
        # Forward: R(+angle) around padded center, then shift origin to rotated center
        fwd_x = np.cos(a) * (px - cx_pad) - np.sin(a) * (py - cy_pad) + cx_rot
        fwd_y = np.sin(a) * (px - cx_pad) + np.cos(a) * (py - cy_pad) + cy_rot

        # Act
        recovered_x, recovered_y = _unrotate_point(
            rotated_point=(fwd_x, fwd_y),
            original_image_size=(self.PADDED_WIDTH, self.PADDED_HEIGHT),
            rotated_image_size=(rot_w, rot_h),
            angle_deg=angle,
        )

        # Assert
        assert recovered_x == pytest.approx(px, abs=1e-6)
        assert recovered_y == pytest.approx(py, abs=1e-6)
