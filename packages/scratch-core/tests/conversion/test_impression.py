import numpy as np
import pytest
from numpy.testing import assert_array_equal

from conversion.data_formats import MarkImage, MarkType, CropType
from conversion.impression import (
    _get_mask_edge_points,
    _points_are_collinear,
    _fit_circle_ransac,
    _get_bounding_box_center,
    _set_map_center,
    _estimate_plane_tilt_degrees,
    _get_valid_coordinates,
    _adjust_for_plane_tilt_degrees,
)


def make_mark_image(data: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0):
    return MarkImage(
        data=data,
        scale_x=scale_x,
        scale_y=scale_y,
        mark_type=MarkType.BREECH_FACE_IMPRESSION,
        crop_type=CropType.RECTANGLE,
    )


class TestGetMaskEdgePoints:
    def test_returns_empty_array_for_empty_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        result = _get_mask_edge_points(mask)
        assert result.shape == (0, 2)

    def test_returns_empty_array_for_single_pixel(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = _get_mask_edge_points(mask)
        # Single pixel erodes to nothing, so edge is the pixel itself
        assert_array_equal(result, [[2, 2]])

    def test_returns_edge_pixels_for_filled_rectangle(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        result = _get_mask_edge_points(mask)
        # 3x3 block: center erodes away, leaving 8 edge pixels
        assert len(result) == 8
        assert [2, 2] not in result.tolist()

    def test_returns_coordinates_as_col_row(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2, 15] = True  # row=2, col=15
        result = _get_mask_edge_points(mask)
        assert_array_equal(result, [[15, 2]])

    def test_returns_only_border_pixels_for_filled_mask(self):
        mask = np.ones((5, 5), dtype=bool)
        result = _get_mask_edge_points(mask)
        assert len(result) == 16
        # Interior pixels should not be present
        interior = {
            (2, 2),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        }
        result_set = {tuple(p) for p in result.tolist()}
        assert result_set.isdisjoint(interior)


class TestPointsAreCollinear:
    def test_returns_true_for_fewer_than_three_points(self):
        assert _points_are_collinear(np.array([[0, 0], [1, 1]]))

    def test_returns_true_for_horizontal_line(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        assert _points_are_collinear(points)

    def test_returns_true_for_vertical_line(self):
        points = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        assert _points_are_collinear(points)

    def test_returns_true_for_diagonal_line(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        assert _points_are_collinear(points)

    def test_returns_false_for_triangle(self):
        points = np.array([[0, 0], [1, 0], [0, 1]])
        assert not _points_are_collinear(points)

    def test_returns_false_for_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert not _points_are_collinear(points)

    def test_returns_true_for_nearly_collinear_within_tolerance(self):
        points = np.array([[0, 0], [1, 1e-12], [2, 0]])
        assert _points_are_collinear(points)

    def test_returns_false_for_nearly_collinear_outside_tolerance(self):
        points = np.array([[0, 0], [1, 0.1], [2, 0]])
        assert not _points_are_collinear(points)


class TestFitCircleRansac:
    def test_returns_none_for_fewer_than_three_points(self):
        points = np.array([[0, 0], [1, 1]])
        assert _fit_circle_ransac(points) is None

    def test_returns_none_for_collinear_points(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        assert _fit_circle_ransac(points) is None

    def test_returns_center_for_perfect_circle(self):
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = np.column_stack([np.cos(theta) + 5, np.sin(theta) + 3])
        center = _fit_circle_ransac(points)
        assert center == pytest.approx((5, 3))

    def test_returns_center_for_circle_with_noise(self):
        rng = np.random.default_rng(42)
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        noise = rng.normal(0, 0.05, size=(100, 2))
        points = np.column_stack([np.cos(theta) + 5, np.sin(theta) + 3]) + noise
        center = _fit_circle_ransac(points)
        assert center == pytest.approx((5, 3), abs=0.1)

    def test_returns_center_for_partial_arc(self):
        theta = np.linspace(0, np.pi / 2, 50)  # quarter circle
        points = np.column_stack([np.cos(theta) * 10, np.sin(theta) * 10])
        center = _fit_circle_ransac(points)
        assert center == pytest.approx((0, 0), abs=0.5)

    def test_raises_for_wrong_shape(self):
        points_1d = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError):
            _fit_circle_ransac(points_1d)

        points_3d = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _fit_circle_ransac(points_3d)


class TestGetBoundingBoxCenter:
    def test_returns_mask_center_for_empty_mask(self):
        mask = np.zeros((10, 20), dtype=bool)
        assert _get_bounding_box_center(mask) == (10, 5)

    def test_returns_pixel_center_for_single_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 7] = True  # row=3, col=7
        assert _get_bounding_box_center(mask) == (7.5, 3.5)

    def test_returns_center_for_rectangle(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 3:9] = True  # rows 2-5, cols 3-8
        assert _get_bounding_box_center(mask) == (6, 4)

    def test_returns_center_for_scattered_pixels(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 2] = True
        mask[7, 8] = True
        assert _get_bounding_box_center(mask) == (5.5, 4.5)

    def test_returns_center_for_filled_mask(self):
        mask = np.ones((6, 10), dtype=bool)
        assert _get_bounding_box_center(mask) == (5, 3)


class TestSetMapCenter:
    def test_uses_bounding_box_when_circle_disabled(self):
        data = np.full((10, 10), np.nan)
        data[2:6, 3:9] = 1.0
        assert _set_map_center(data, use_circle=False) == (6, 4)

    def test_falls_back_to_bounding_box_for_collinear_edge(self):
        data = np.full((10, 10), np.nan)
        data[5, :] = 1.0
        assert _set_map_center(data, use_circle=True) == (5, 5.5)

    def test_returns_array_center_for_all_nan(self):
        data = np.full((10, 20), np.nan)
        assert _set_map_center(data) == (10, 5)


class TestEstimatePlaneTiltDegrees:
    def test_returns_zero_tilt_for_flat_plane(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = np.array([5, 5, 5, 5, 5, 5])
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z)
        assert tilt_x == pytest.approx(0)
        assert tilt_y == pytest.approx(0)
        assert residuals == pytest.approx(np.zeros(6))

    def test_returns_45_degree_tilt_x(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x.astype(float)  # z = x, slope of 1
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z)
        assert tilt_x == pytest.approx(45)
        assert tilt_y == pytest.approx(0)
        assert residuals == pytest.approx(np.zeros(6))

    def test_returns_45_degree_tilt_y(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = y.astype(float)  # z = y, slope of 1
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z)
        assert tilt_x == pytest.approx(0)
        assert tilt_y == pytest.approx(45)
        assert residuals == pytest.approx(np.zeros(6))

    def test_returns_negative_tilt(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = -x.astype(float)
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z)
        assert tilt_x == pytest.approx(-45)
        assert tilt_y == pytest.approx(0)

    def test_returns_combined_tilt(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x + y  # z = x + y, slope of 1 in both directions
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z.astype(float))
        assert tilt_x == pytest.approx(45)
        assert tilt_y == pytest.approx(45)
        assert residuals == pytest.approx(np.zeros(6))

    def test_returns_residuals_for_noisy_data(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x + y + np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z)
        assert tilt_x == pytest.approx(45, abs=1)
        assert tilt_y == pytest.approx(45, abs=1)
        assert not np.allclose(residuals, 0)

    def test_returns_small_tilt_angle(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = 0.1 * x  # slope of 0.1
        tilt_x, tilt_y, residuals = _estimate_plane_tilt_degrees(x, y, z.astype(float))
        assert tilt_x == pytest.approx(np.degrees(np.arctan(0.1)))
        assert tilt_y == pytest.approx(0)


class TestGetValidCoordinates:
    def test_returns_empty_arrays_for_all_nan(self):
        mark_image = make_mark_image(
            data=np.full((5, 5), np.nan),
            scale_x=1.0,
            scale_y=1.0,
        )
        x, y, z = _get_valid_coordinates(mark_image, center=(0, 0))
        assert len(x) == 0
        assert len(y) == 0
        assert len(z) == 0

    def test_returns_coordinates_for_single_pixel(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0  # row=2, col=3
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(mark_image, center=(0, 0))
        assert x == pytest.approx([3.0])
        assert y == pytest.approx([2.0])
        assert z == pytest.approx([10.0])

    def test_applies_scale_factors(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0
        mark_image = make_mark_image(data=data, scale_x=0.5, scale_y=0.25)
        x, y, z = _get_valid_coordinates(mark_image, center=(0, 0))
        assert x == pytest.approx([1.5])  # 3 * 0.5
        assert y == pytest.approx([0.5])  # 2 * 0.25
        assert z == pytest.approx([10.0])

    def test_subtracts_center(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(mark_image, center=(1.0, 0.5))
        assert x == pytest.approx([2.0])  # 3 - 1.0
        assert y == pytest.approx([1.5])  # 2 - 0.5
        assert z == pytest.approx([10.0])

    def test_returns_all_valid_pixels(self):
        data = np.full((3, 3), np.nan)
        data[0, 0] = 1.0
        data[1, 2] = 2.0
        data[2, 1] = 3.0
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(mark_image, center=(0, 0))
        assert len(x) == 3
        assert set(z) == {1.0, 2.0, 3.0}


class TestAdjustForPlaneTiltDegrees:
    def test_raises_for_fewer_than_three_valid_points(self):
        data = np.full((5, 5), np.nan)
        data[0, 0] = 1.0
        data[1, 1] = 2.0
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        with pytest.raises(ValueError):
            _adjust_for_plane_tilt_degrees(mark_image, center=(0, 0))

    def test_returns_flat_data_for_tilted_plane(self):
        data = np.full((5, 5), np.nan)
        for row in range(5):
            for col in range(5):
                data[row, col] = float(col)  # tilted in x direction
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        result = _adjust_for_plane_tilt_degrees(mark_image, center=(0, 0))
        valid_mask = ~np.isnan(result.data)
        assert result.data[valid_mask] == pytest.approx(np.zeros(25), abs=1e-10)

    def test_adjusts_scale_for_tilt(self):
        data = np.full((5, 5), np.nan)
        for row in range(5):
            for col in range(5):
                data[row, col] = float(col)
        mark_image = make_mark_image(data=data, scale_x=1.0, scale_y=1.0)
        result = _adjust_for_plane_tilt_degrees(mark_image, center=(0, 0))
        assert result.scale_x > mark_image.scale_x
        assert result.scale_y == pytest.approx(mark_image.scale_y)
