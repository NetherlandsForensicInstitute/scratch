import numpy as np
import pytest
from skimage.transform import rotate
from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.coarse import (
    match_cells,
    _get_fill_fraction_map,
    _get_score_map,
    _compute_rotation_center,
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

# Constants used by _compute_rotation_center and _unrotate_point tests
PADDED_WIDTH = 160
PADDED_HEIGHT = 120
PAD_WIDTH = 20
PAD_HEIGHT = 20
PADDED_CENTER = (PADDED_WIDTH / 2, PADDED_HEIGHT / 2)


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


class TestComputeRotationCenter:
    def test_angle_zero_returns_padded_center(self):
        # Arrange / Act
        cx, cy = _compute_rotation_center(
            padded_size=(PADDED_WIDTH, PADDED_HEIGHT), angle=0.0
        )

        # Assert — at zero rotation the center is unchanged
        assert cx == pytest.approx(PADDED_WIDTH / 2)
        assert cy == pytest.approx(PADDED_HEIGHT / 2)

    @pytest.mark.parametrize("angle", [30.0, 60.0, 90.0, 145.0])
    def test_rotation_center_lies_within_output_image(self, angle: float):
        # Arrange
        padded = np.zeros((PADDED_HEIGHT, PADDED_WIDTH))
        rotated = rotate(padded, angle=-angle, cval=0, order=0, resize=True)
        out_h, out_w = rotated.shape  # type: ignore

        # Act
        cx, cy = _compute_rotation_center(
            padded_size=(PADDED_WIDTH, PADDED_HEIGHT), angle=angle
        )

        # Assert — the output rotation center must be inside the rotated canvas
        assert 0.0 <= cx <= out_w
        assert 0.0 <= cy <= out_h


class TestUnrotatePoint:
    def test_angle_zero_removes_padding_only(self):
        # Arrange
        rotation_center = _compute_rotation_center(
            padded_size=(PADDED_WIDTH, PADDED_HEIGHT), angle=0.0
        )
        rotated_point = (90.0, 70.0)  # arbitrary point in the rotated (=padded) image

        # Act
        x, y = _unrotate_point(
            rotated_point=rotated_point,
            angle=0.0,  # at angle=0 unrotate just strips padding
            pad_size=(PAD_WIDTH, PAD_HEIGHT),
            padded_center=PADDED_CENTER,
            rotation_center=rotation_center,
        )

        # Assert
        assert x == pytest.approx(rotated_point[0] - PAD_WIDTH)
        assert y == pytest.approx(rotated_point[1] - PAD_HEIGHT)

    @pytest.mark.parametrize("angle", [45.0, 60.0, 120.0])
    def test_round_trip_recovers_original_comparison_coordinate(self, angle: float):
        # Arrange — construct the forward-rotated position analytically from a known
        # original comparison coordinate, then verify unrotate recovers it exactly.
        original_x, original_y = 60.0, 40.0
        padded_x = original_x + PAD_WIDTH
        padded_y = original_y + PAD_HEIGHT

        rotation_center = _compute_rotation_center(
            padded_size=(PADDED_WIDTH, PADDED_HEIGHT), angle=angle
        )
        cx_rc, cy_rc = rotation_center
        cx_pc, cy_pc = PADDED_CENTER

        # Forward map: rotated = R(-angle)*(padded - padded_center) + rotation_center
        a = np.radians(-angle)
        rotated_x = (
            np.cos(a) * (padded_x - cx_pc) - np.sin(a) * (padded_y - cy_pc) + cx_rc
        )
        rotated_y = (
            np.sin(a) * (padded_x - cx_pc) + np.cos(a) * (padded_y - cy_pc) + cy_rc
        )

        # Act
        recovered_x, recovered_y = _unrotate_point(
            rotated_point=(rotated_x, rotated_y),
            angle=angle,
            pad_size=(PAD_WIDTH, PAD_HEIGHT),
            padded_center=PADDED_CENTER,
            rotation_center=rotation_center,
        )

        # Assert
        assert recovered_x == pytest.approx(original_x, abs=1e-6)
        assert recovered_y == pytest.approx(original_y, abs=1e-6)
