from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, Mark, MarkType
from conversion.impression import (
    _adjust_for_plane_tilt,
    _apply_anti_aliasing,
    _build_preprocessing_metadata,
    _compute_center_local,
    _compute_map_center,
    _estimate_plane_tilt,
    _fit_circle_ransac,
    _get_bounding_box_center,
    _get_mask_edge_points,
    _get_valid_coordinates,
    _needs_resampling,
    _points_are_collinear,
    _update_mark_data,
    preprocess_impression_mark,
)
from conversion.parameters import PreprocessingImpressionParams
from tests.conversion.helper_function import (
    _crop_to_common_shape,
    _compute_correlation,
    _compute_difference_stats,
)


def make_circular_data(
    shape: tuple[int, int], center: tuple[float, float], radius: float
) -> np.ndarray:
    """Create circular height map data with NaN outside circle."""
    data = np.full(shape, np.nan)
    y, x = np.ogrid[: shape[0], : shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
    # Add some height variation (dome shape)
    data[mask] = radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)[mask]
    return data


def make_rectangular_data(
    shape: tuple[int, int], margin: int = 10, scale: float = 1e-6
) -> np.ndarray:
    """Create rectangular height map data with NaN border.

    :param shape: Shape of the output array.
    :param margin: Size of NaN border.
    :param scale: Scale factor for height values (should match pixel scale).
    :return: Height map with tilted plane and noise.
    """
    data = np.full(shape, np.nan)
    # Add tilted plane with some noise, scaled appropriately
    y, x = np.mgrid[: shape[0], : shape[1]]
    data[margin:-margin, margin:-margin] = (
        0.1 * x[margin:-margin, margin:-margin] * scale
        + 0.05 * y[margin:-margin, margin:-margin] * scale
        + np.random.normal(
            0, 0.01 * scale, (shape[0] - 2 * margin, shape[1] - 2 * margin)
        )
    )
    return data


def make_mark(
    data: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    mark_type: MarkType = MarkType.EXTRACTOR_IMPRESSION,
) -> Mark:
    """Create a Mark instance for testing."""
    return Mark(
        scan_image=ScanImage(
            data=data,
            scale_x=scale_x,
            scale_y=scale_y,
        ),
        mark_type=mark_type,
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


class TestComputeMapCenter:
    def test_uses_bounding_box_when_circle_disabled(self):
        data = np.full((10, 10), np.nan)
        data[2:6, 3:9] = 1.0
        assert _compute_map_center(data, use_circle_fit=False) == (6, 4)

    def test_falls_back_to_bounding_box_for_collinear_edge(self):
        data = np.full((10, 10), np.nan)
        data[5, :] = 1.0
        assert _compute_map_center(data, use_circle_fit=True) == (5, 5.5)

    def test_returns_array_center_for_all_nan(self):
        data = np.full((10, 20), np.nan)
        assert _compute_map_center(data) == (10, 5)


class TestEstimatePlaneTilt:
    def test_returns_zero_tilt_for_flat_plane(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = np.array([5, 5, 5, 5, 5, 5])
        result = _estimate_plane_tilt(x, y, z)
        assert result.tilt_x_deg == pytest.approx(0)
        assert result.tilt_y_deg == pytest.approx(0)
        assert result.residuals == pytest.approx(np.zeros(6))

    def test_returns_45_degree_tilt_x(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x.astype(float)  # z = x, slope of 1
        result = _estimate_plane_tilt(x, y, z)
        assert result.tilt_x_deg == pytest.approx(45)
        assert result.tilt_y_deg == pytest.approx(0)
        assert result.residuals == pytest.approx(np.zeros(6))

    def test_returns_45_degree_tilt_y(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = y.astype(float)  # z = y, slope of 1
        result = _estimate_plane_tilt(x, y, z)
        assert result.tilt_x_deg == pytest.approx(0)
        assert result.tilt_y_deg == pytest.approx(45)
        assert result.residuals == pytest.approx(np.zeros(6))

    def test_returns_negative_tilt(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = -x.astype(float)
        result = _estimate_plane_tilt(x, y, z)
        assert result.tilt_x_deg == pytest.approx(-45)
        assert result.tilt_y_deg == pytest.approx(0)

    def test_returns_combined_tilt(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x + y  # z = x + y, slope of 1 in both directions
        result = _estimate_plane_tilt(x, y, z.astype(float))
        assert result.tilt_x_deg == pytest.approx(45)
        assert result.tilt_y_deg == pytest.approx(45)
        assert result.residuals == pytest.approx(np.zeros(6))

    def test_returns_residuals_for_noisy_data(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = x + y + np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
        result = _estimate_plane_tilt(x, y, z)
        assert result.tilt_x_deg == pytest.approx(45, abs=1)
        assert result.tilt_y_deg == pytest.approx(45, abs=1)
        assert not np.allclose(result.residuals, 0)

    def test_returns_small_tilt_angle(self):
        x = np.array([0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1])
        z = 0.1 * x  # slope of 0.1
        result = _estimate_plane_tilt(x, y, z.astype(float))
        assert result.tilt_x_deg == pytest.approx(np.degrees(np.arctan(0.1)))
        assert result.tilt_y_deg == pytest.approx(0)


class TestGetValidCoordinates:
    def test_returns_empty_arrays_for_all_nan(self):
        scan_image = ScanImage(
            data=np.full((5, 5), np.nan),
            scale_x=1.0,
            scale_y=1.0,
        )
        x, y, z = _get_valid_coordinates(scan_image, center=(0, 0))
        assert len(x) == 0
        assert len(y) == 0
        assert len(z) == 0

    def test_returns_coordinates_for_single_pixel(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0  # row=2, col=3
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(scan_image, center=(0, 0))
        assert x == pytest.approx([3.0])
        assert y == pytest.approx([2.0])
        assert z == pytest.approx([10.0])

    def test_applies_scale_factors(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0
        scan_image = ScanImage(data=data, scale_x=0.5, scale_y=0.25)
        x, y, z = _get_valid_coordinates(scan_image, center=(0, 0))
        assert x == pytest.approx([1.5])  # 3 * 0.5
        assert y == pytest.approx([0.5])  # 2 * 0.25
        assert z == pytest.approx([10.0])

    def test_subtracts_center(self):
        data = np.full((5, 5), np.nan)
        data[2, 3] = 10.0
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(scan_image, center=(1.0, 0.5))
        assert x == pytest.approx([2.0])  # 3 - 1.0
        assert y == pytest.approx([1.5])  # 2 - 0.5
        assert z == pytest.approx([10.0])

    def test_returns_all_valid_pixels(self):
        data = np.full((3, 3), np.nan)
        data[0, 0] = 1.0
        data[1, 2] = 2.0
        data[2, 1] = 3.0
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        x, y, z = _get_valid_coordinates(scan_image, center=(0, 0))
        assert len(x) == 3
        assert set(z) == {1.0, 2.0, 3.0}


class TestAdjustForPlaneTilt:
    def test_raises_for_fewer_than_three_valid_points(self):
        data = np.full((5, 5), np.nan)
        data[0, 0] = 1.0
        data[1, 1] = 2.0
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        with pytest.raises(ValueError):
            _adjust_for_plane_tilt(scan_image, center=(0, 0))

    def test_returns_flat_data_for_tilted_plane(self):
        data = np.full((5, 5), np.nan)
        for row in range(5):
            for col in range(5):
                data[row, col] = float(col)  # tilted in x direction
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        result, _ = _adjust_for_plane_tilt(scan_image, center=(0, 0))
        valid_mask = ~np.isnan(result.data)
        assert result.data[valid_mask] == pytest.approx(np.zeros(25), abs=1e-10)

    def test_adjusts_scale_for_tilt(self):
        data = np.full((5, 5), np.nan)
        for row in range(5):
            for col in range(5):
                data[row, col] = float(col)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        result, _ = _adjust_for_plane_tilt(scan_image, center=(0, 0))
        assert result.scale_x > scan_image.scale_x
        assert result.scale_y == pytest.approx(scan_image.scale_y)


class TestApplyAntiAliasing:
    def test_returns_original_when_below_threshold(self):
        data = np.random.default_rng(42).random((10, 10))
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        result, cutoffs = _apply_anti_aliasing(scan_image, target_spacing=(1.4, 1.4))
        assert result is scan_image
        assert cutoffs == (None, None)

    def test_applies_filter_when_above_threshold(self):
        data = np.random.default_rng(42).random((10, 10))
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)
        _, cutoffs = _apply_anti_aliasing(scan_image, target_spacing=(2.0, 2.0))
        assert cutoffs == (2.0, 2.0)


class TestComputeCenterLocal:
    def test_returns_center_in_meters(self):
        """Center should be converted to meters using scale."""
        data = np.full((10, 10), np.nan)
        data[2:8, 2:8] = 1.0
        mark = make_mark(data, scale_x=1e-6, scale_y=2e-6)
        center_local = _compute_center_local(mark)

        # Pixel center is (5, 5), scaled: (5 * 1e-6, 5 * 2e-6)
        assert center_local[0] == pytest.approx(5e-6)
        assert center_local[1] == pytest.approx(10e-6)

    def test_breech_face_uses_circle_fitting(self):
        """Breech face impression should use circle fitting."""
        data = np.full((41, 41), np.nan)
        center_true = (20, 20)
        radius = 15
        y, x = np.ogrid[:41, :41]
        circle_mask = (x - center_true[0]) ** 2 + (y - center_true[1]) ** 2 <= radius**2
        data[circle_mask] = 1.0
        mark = make_mark(
            data, scale_x=1e-6, scale_y=1e-6, mark_type=MarkType.BREECH_FACE_IMPRESSION
        )

        center_local = _compute_center_local(mark)

        # Should be close to (20, 20) in pixels -> (20e-6, 20e-6) in meters
        assert center_local[0] == pytest.approx(20e-6)
        assert center_local[1] == pytest.approx(20e-6)


class TestPreprocessImpressionMarkIntegration:
    """Integration tests for preprocess_impression_mark."""

    @pytest.mark.integration
    def test_basic_pipeline_runs(self):
        """Basic pipeline should run without errors."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
            level_offset=True,
            level_tilt=True,
            level_2nd=False,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_output_has_correct_scale(self):
        """Output should have the target pixel size."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        target_size = (2e-6, 2e-6)
        params = PreprocessingImpressionParams(
            pixel_size=target_size,
            adjust_pixel_spacing=False,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert filtered.scan_image.scale_x == target_size[0]
        assert filtered.scan_image.scale_y == target_size[1]
        assert leveled.scan_image.scale_x == target_size[0]
        assert leveled.scan_image.scale_y == target_size[1]

    @pytest.mark.integration
    def test_output_is_smaller_after_downsampling(self):
        """Downsampling should reduce array size."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),  # 2x downsampling
            adjust_pixel_spacing=False,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        # After 2x downsampling, size should be roughly half
        assert filtered.scan_image.data.shape[0] < mark.scan_image.data.shape[0]
        assert filtered.scan_image.data.shape[1] < mark.scan_image.data.shape[1]
        assert leveled.scan_image.data.shape[0] < mark.scan_image.data.shape[0]
        assert leveled.scan_image.data.shape[1] < mark.scan_image.data.shape[1]

    @pytest.mark.integration
    def test_metadata_is_populated(self):
        """Output metadata should contain processing info."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        # Check filtered metadata
        assert filtered.meta_data.get("is_filtered") is True
        assert filtered.meta_data.get("is_leveled") is True
        assert filtered.meta_data.get("is_prep") is True
        assert "center_l_x" in filtered.meta_data
        assert "center_l_y" in filtered.meta_data

        # Check leveled metadata
        assert leveled.meta_data.get("is_filtered") is False
        assert leveled.meta_data.get("is_leveled") is True

    @pytest.mark.integration
    def test_filtered_and_leveled_differ(self):
        """Filtered and leveled outputs should be different."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            adjust_pixel_spacing=False,
            highpass_cutoff=50e-6,  # Apply high-pass to create difference
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        # Data should differ due to high-pass filtering
        assert not np.allclose(
            filtered.scan_image.data, leveled.scan_image.data, equal_nan=True
        )

    @pytest.mark.integration
    def test_breech_face_uses_circle_center(self):
        """Breech face impression should use circle fitting for center."""
        center = (50, 50)
        data = make_circular_data((100, 100), center, radius=40)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(1e-6, 1e-6),  # No resampling
            adjust_pixel_spacing=False,
        )

        filtered, _ = preprocess_impression_mark(mark, params)

        # Center should be close to (50, 50) pixels = (50e-6, 50e-6) meters
        center_x = filtered.meta_data.get("center_l_x")
        center_y = filtered.meta_data.get("center_l_y")
        assert center_x == pytest.approx(50e-6)
        assert center_y == pytest.approx(50e-6)

    @pytest.mark.integration
    def test_no_resampling_when_pixel_size_matches(self):
        """No resampling should occur when pixel size already matches."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(1e-6, 1e-6),  # Same as input
            adjust_pixel_spacing=False,
        )

        filtered, _ = preprocess_impression_mark(mark, params)

        assert filtered.meta_data.get("is_interpolated") is False

    @pytest.mark.integration
    def test_interpolated_flag_set_on_resampling(self):
        """Interpolated flag should be True after resampling."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),  # Different from input
            adjust_pixel_spacing=False,
        )

        filtered, _ = preprocess_impression_mark(mark, params)

        assert filtered.meta_data.get("is_interpolated") is True

    @pytest.mark.integration
    def test_without_lowpass_filter(self):
        """Pipeline should work without low-pass filter."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
            lowpass_cutoff=None,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_without_highpass_filter(self):
        """Pipeline should work without high-pass filter."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
            highpass_cutoff=None,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_without_any_filters(self):
        """Pipeline should work without any filters."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
            lowpass_cutoff=None,
            highpass_cutoff=None,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_with_tilt_adjustment(self):
        """Pipeline should work with tilt adjustment enabled."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=True,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_with_second_order_leveling(self):
        """Pipeline should work with second order leveling."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
            level_offset=True,
            level_tilt=True,
            level_2nd=True,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        assert isinstance(filtered, Mark)
        assert isinstance(leveled, Mark)

    @pytest.mark.integration
    def test_output_data_is_finite_where_valid(self):
        """Output data should be finite where input was valid."""
        data = make_rectangular_data((100, 100), scale=1e-6)
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(2e-6, 2e-6),
            adjust_pixel_spacing=False,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        # Valid pixels should be finite
        filtered_valid = ~np.isnan(filtered.scan_image.data)
        leveled_valid = ~np.isnan(leveled.scan_image.data)

        assert np.all(np.isfinite(filtered.scan_image.data[filtered_valid]))
        assert np.all(np.isfinite(leveled.scan_image.data[leveled_valid]))

    @pytest.mark.integration
    def test_leveled_preserves_form(self):
        """Leveled output should preserve higher-order form (only rigid leveling)."""
        # Create data with curvature (sphere-like)
        data = np.full((100, 100), np.nan)
        y, x = np.mgrid[:100, :100]
        center = (50, 50)
        # Add parabolic form + tilt
        data[10:-10, 10:-10] = (
            0.001
            * (
                (x[10:-10, 10:-10] - center[0]) ** 2
                + (y[10:-10, 10:-10] - center[1]) ** 2
            )
            + 0.1 * x[10:-10, 10:-10]
            + 0.05 * y[10:-10, 10:-10]
        )
        mark = make_mark(
            data=data,
            scale_x=1e-6,
            scale_y=1e-6,
            mark_type=MarkType.FIRING_PIN_IMPRESSION,
        )
        params = PreprocessingImpressionParams(
            pixel_size=(1e-6, 1e-6),
            adjust_pixel_spacing=False,
            level_offset=True,
            level_tilt=True,
            level_2nd=True,  # Remove curvature from filtered
            lowpass_cutoff=None,
            highpass_cutoff=None,
        )

        filtered, leveled = preprocess_impression_mark(mark, params)

        # Leveled should have more variance (curvature preserved)
        # Filtered should have less variance (curvature removed)
        filtered_var = np.nanvar(filtered.scan_image.data)
        leveled_var = np.nanvar(leveled.scan_image.data)

        # Leveled should preserve more of the original form
        assert leveled_var > filtered_var


class TestUpdateMarkData:
    """Tests for _update_mark_data helper."""

    def test_returns_new_mark_with_updated_data(self):
        """Should return a new Mark instance with updated data."""
        original_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mark = make_mark(original_data, scale_x=1.0, scale_y=1.0)

        new_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = _update_mark_data(mark, new_data)

        assert_array_equal(result.scan_image.data, new_data)

    def test_does_not_modify_original_mark(self):
        """Original mark should remain unchanged."""
        original_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mark = make_mark(original_data, scale_x=1.0, scale_y=1.0)

        new_data = np.array([[5.0, 6.0], [7.0, 8.0]])
        _update_mark_data(mark, new_data)

        assert_array_equal(mark.scan_image.data, original_data)

    def test_preserves_scale_factors(self):
        """Scale factors should be preserved."""
        mark = make_mark(np.zeros((3, 3)), scale_x=0.5, scale_y=0.25)

        result = _update_mark_data(mark, np.ones((3, 3)))

        assert result.scan_image.scale_x == 0.5
        assert result.scan_image.scale_y == 0.25

    def test_preserves_mark_type(self):
        """Mark type should be preserved."""
        mark = make_mark(
            np.zeros((3, 3)),
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
        )

        result = _update_mark_data(mark, np.ones((3, 3)))

        assert result.mark_type == MarkType.BREECH_FACE_IMPRESSION

    def test_preserves_crop_type(self):
        """Crop type should be preserved."""
        mark = make_mark(np.zeros((3, 3)))

        result = _update_mark_data(mark, np.ones((3, 3)))

        assert result.crop_type == CropType.RECTANGLE

    def test_handles_nan_values(self):
        """Should handle NaN values in data."""
        mark = make_mark(np.zeros((3, 3)))
        new_data = np.array(
            [[1.0, np.nan, 2.0], [np.nan, 3.0, np.nan], [4.0, 5.0, 6.0]]
        )

        result = _update_mark_data(mark, new_data)

        assert np.isnan(result.scan_image.data[0, 1])
        assert np.isnan(result.scan_image.data[1, 0])
        assert result.scan_image.data[0, 0] == 1.0


class TestNeedsResampling:
    """Tests for _needs_resampling."""

    def test_returns_false_when_scales_match_exactly(self):
        """Should return False when scales match exactly."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1.0, scale_y=1.0)

        assert _needs_resampling(mark, target_scale=1.0) is False

    def test_returns_false_when_scales_match_within_tolerance(self):
        """Should return False when scales match within relative tolerance."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1.0, scale_y=1.0 + 1e-9)

        assert _needs_resampling(mark, target_scale=1.0) is False

    def test_returns_true_when_scales_differ(self):
        """Should return True when scales differ significantly."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1.0, scale_y=1.0)

        assert _needs_resampling(mark, target_scale=2.0) is True

    def test_returns_true_when_only_x_differs(self):
        """Should return True when only x scale differs."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1.5, scale_y=1.0)

        assert _needs_resampling(mark, target_scale=1.0) is True

    def test_returns_true_when_only_y_differs(self):
        """Should return True when only y scale differs."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1.0, scale_y=1.5)

        assert _needs_resampling(mark, target_scale=1.0) is True

    def test_handles_small_scale_values(self):
        """Should handle small scale values correctly."""
        mark = make_mark(np.zeros((10, 10)), scale_x=1e-6, scale_y=1e-6)

        assert _needs_resampling(mark, target_scale=1e-6) is False
        assert _needs_resampling(mark, target_scale=2e-6) is True


class TestBuildPreprocessingMetadata:
    """Tests for _build_preprocessing_metadata."""

    def test_includes_params_fields(self):
        """Should include all preprocessing params fields."""
        params = PreprocessingImpressionParams(
            pixel_size=(1e-6, 1e-6),
            adjust_pixel_spacing=True,
            lowpass_cutoff=10e-6,
            highpass_cutoff=50e-6,
        )
        center_local = (5e-6, 10e-6)

        result = _build_preprocessing_metadata(params, center_local, interpolated=True)

        assert result["pixel_size"] == (1e-6, 1e-6)
        assert result["adjust_pixel_spacing"] is True
        assert result["lowpass_cutoff"] == 10e-6
        assert result["highpass_cutoff"] == 50e-6

    def test_includes_center_coordinates(self):
        """Should include local center coordinates."""
        params = PreprocessingImpressionParams(pixel_size=(1e-6, 1e-6))
        center_local = (5e-6, 10e-6)

        result = _build_preprocessing_metadata(params, center_local, interpolated=False)

        assert result["center_l_x"] == 5e-6
        assert result["center_l_y"] == 10e-6

    def test_includes_global_center_as_zero(self):
        """Should include global center as zero."""
        params = PreprocessingImpressionParams(pixel_size=(1e-6, 1e-6))

        result = _build_preprocessing_metadata(params, (0, 0), interpolated=False)

        assert result["center_g_x"] == 0
        assert result["center_g_y"] == 0

    def test_includes_processing_flags(self):
        """Should include processing flags."""
        params = PreprocessingImpressionParams(pixel_size=(1e-6, 1e-6))

        result = _build_preprocessing_metadata(params, (0, 0), interpolated=True)

        assert result["is_crop"] is True
        assert result["is_prep"] is True
        assert result["is_interpolated"] is True

    def test_interpolated_flag_false(self):
        """Should set interpolated flag to False when specified."""
        params = PreprocessingImpressionParams(pixel_size=(1e-6, 1e-6))

        result = _build_preprocessing_metadata(params, (0, 0), interpolated=False)

        assert result["is_interpolated"] is False


class TestMatlabPythonComparison:
    """Compare Python preprocessing output against MATLAB reference output."""

    @pytest.fixture
    def input_mark(self, baseline_images_dir: Path) -> "Mark":
        """Load input used for MATLAB test."""
        data = np.load(
            baseline_images_dir / "preprocessing_impression_input.npy",
            allow_pickle=True,
        )
        scan_image = ScanImage(data=data, scale_x=3.5000e-06, scale_y=3.5000e-06)
        return Mark(
            scan_image=scan_image,
            mark_type=MarkType.BREECH_FACE_IMPRESSION,
            crop_type=CropType.RECTANGLE,
        )

    @pytest.fixture
    def matlab_outputs(self, baseline_images_dir: Path):
        """Load MATLAB reference outputs.

        Override this fixture to provide paths to your MATLAB output files.
        Expected to return dict with 'processed' and 'leveled' arrays.
        """
        matlab_processed_path = Path(
            "preprocessing_impression_output_matlab_processed.npy"
        )
        matlab_leveled_path = Path("preprocessing_impression_output_matlab_leveled.npy")

        return {
            "processed": np.load(baseline_images_dir / matlab_processed_path),
            "leveled": np.load(baseline_images_dir / matlab_leveled_path),
        }

    @pytest.fixture
    def python_outputs(self, input_mark: Mark):
        """Generate Python preprocessing outputs.

        Override this fixture to load your test input and run preprocessing.
        """
        params = PreprocessingImpressionParams()
        processed, leveled = preprocess_impression_mark(input_mark, params)
        return {
            "processed": processed.scan_image.data,
            "leveled": leveled.scan_image.data,
        }

    def test_processed_output_correlation(self, matlab_outputs, python_outputs):
        """Test that processed outputs have high correlation (>0.999)."""
        matlab_proc, python_proc = _crop_to_common_shape(
            matlab_outputs["processed"], python_outputs["processed"]
        )

        correlation = _compute_correlation(matlab_proc, python_proc)

        assert correlation > 0.999, (
            f"Processed correlation {correlation} below threshold"
        )

    def test_processed_output_range(self, matlab_outputs, python_outputs):
        """Test that processed outputs have similar value ranges."""
        matlab_proc = matlab_outputs["processed"]
        python_proc = python_outputs["processed"]

        matlab_range = (np.nanmin(matlab_proc), np.nanmax(matlab_proc))
        python_range = (np.nanmin(python_proc), np.nanmax(python_proc))

        # Check ranges are within 10% of each other
        matlab_span = matlab_range[1] - matlab_range[0]
        python_span = python_range[1] - python_range[0]

        range_ratio = python_span / matlab_span if matlab_span != 0 else np.inf

        assert 0.9 < range_ratio < 1.1, (
            f"Range mismatch: MATLAB {matlab_range}, Python {python_range}"
        )

    def test_processed_output_difference(self, matlab_outputs, python_outputs):
        """Test that processed outputs have small absolute differences."""
        matlab_proc, python_proc = _crop_to_common_shape(
            matlab_outputs["processed"], python_outputs["processed"]
        )

        stats = _compute_difference_stats(python_proc, matlab_proc)

        # Mean difference should be near zero
        assert abs(stats["mean"]) < 1e-9, f"Mean difference {stats['mean']} too large"

        # Std of differences should be small relative to signal
        signal_std = np.nanstd(matlab_proc)
        relative_std = stats["std"] / signal_std if signal_std != 0 else np.inf

        assert relative_std < 0.01, f"Relative std {relative_std} too large"

    def test_leveled_output_correlation(self, matlab_outputs, python_outputs):
        """Test that leveled outputs have high correlation (>0.999)."""
        matlab_lev, python_lev = _crop_to_common_shape(
            matlab_outputs["leveled"], python_outputs["leveled"]
        )

        correlation = _compute_correlation(matlab_lev, python_lev)

        assert correlation > 0.999, f"Leveled correlation {correlation} below threshold"

    def test_leveled_output_range(self, matlab_outputs, python_outputs):
        """Test that leveled outputs have similar value ranges."""
        matlab_lev = matlab_outputs["leveled"]
        python_lev = python_outputs["leveled"]

        matlab_range = (np.nanmin(matlab_lev), np.nanmax(matlab_lev))
        python_range = (np.nanmin(python_lev), np.nanmax(python_lev))

        # Check ranges are within 10% of each other
        matlab_span = matlab_range[1] - matlab_range[0]
        python_span = python_range[1] - python_range[0]

        range_ratio = python_span / matlab_span if matlab_span != 0 else np.inf

        assert 0.9 < range_ratio < 1.1, (
            f"Range mismatch: MATLAB {matlab_range}, Python {python_range}"
        )

    def test_leveled_output_difference(self, matlab_outputs, python_outputs):
        """Test that leveled outputs have small absolute differences."""
        matlab_lev, python_lev = _crop_to_common_shape(
            matlab_outputs["leveled"], python_outputs["leveled"]
        )
        stats = _compute_difference_stats(python_lev, matlab_lev)

        # Mean difference should be near zero
        assert abs(stats["mean"]) < 1e-9, f"Mean difference {stats['mean']} too large"

        # Std of differences should be small relative to signal
        signal_std = np.nanstd(matlab_lev)
        relative_std = stats["std"] / signal_std if signal_std != 0 else np.inf

        assert relative_std < 0.01, f"Relative std {relative_std} too large"

    def test_shape_similarity(self, matlab_outputs, python_outputs):
        """Test that output shapes are within 1 pixel of each other."""
        matlab_proc_shape = matlab_outputs["processed"].shape
        python_proc_shape = python_outputs["processed"].shape

        row_diff = abs(matlab_proc_shape[0] - python_proc_shape[0])
        col_diff = abs(matlab_proc_shape[1] - python_proc_shape[1])

        assert row_diff <= 1, f"Row difference {row_diff} too large"
        assert col_diff <= 1, f"Column difference {col_diff} too large"

    def test_nan_pattern_similarity(self, matlab_outputs, python_outputs):
        """Test that NaN patterns are similar between outputs."""
        matlab_proc, python_proc = _crop_to_common_shape(
            matlab_outputs["processed"], python_outputs["processed"]
        )

        matlab_nan = np.isnan(matlab_proc)
        python_nan = np.isnan(python_proc)

        # Calculate percentage of matching NaN positions
        matching = np.sum(matlab_nan == python_nan)
        total = matlab_nan.size
        match_ratio = matching / total

        assert match_ratio > 0.99, f"NaN pattern match ratio {match_ratio} too low"
