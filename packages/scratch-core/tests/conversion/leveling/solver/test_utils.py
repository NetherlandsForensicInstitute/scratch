import numpy as np
import pytest

from conversion.leveling.solver import compute_root_mean_square
from conversion.leveling.solver.utils import compute_image_center
from image_generation.data_formats import ScanImage


class TestRootMeanSquare:
    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0, -3.15, 40.123, -80, 100])
    def test_rms_is_constant_for_constant_input(self, value: float):
        result = compute_root_mean_square(np.array([value] * 100))
        assert np.isclose(result, abs(value))

    @pytest.mark.parametrize("value", list(range(-10, 10)))
    def test_rms_is_non_negative(self, value: float):
        result = compute_root_mean_square(np.array([value] * 100))
        assert result > 0 or np.isclose(result, 0.0)

    def test_rms_can_handle_nans(self):
        array_with_nans = np.array(
            [0, 1, 0.15, 2, np.nan, 3, 4, np.nan, -5], dtype=np.float64
        )
        array_without_nans = array_with_nans[~np.isnan(array_with_nans)]

        result_with_nans = compute_root_mean_square(array_with_nans)
        result_without_nans = compute_root_mean_square(array_without_nans)
        assert np.isclose(result_with_nans, result_without_nans)
        assert np.isclose(result_with_nans, 2.8036328473709147)


class TestComputeImageCenter:
    def test_compute_image_center_for_square_image(
        self, scan_image_with_nans: ScanImage
    ):
        center_x, center_y = compute_image_center(scan_image_with_nans)
        assert np.isclose(center_x, 0.0004037151235)
        assert np.isclose(center_y, center_x)

    def test_compute_image_center_for_rectangular_image(
        self, scan_image_rectangular_with_nans: ScanImage
    ):
        center_x, center_y = compute_image_center(scan_image_rectangular_with_nans)
        assert np.isclose(center_x, 0.00030245829675)
        assert np.isclose(center_y, 0.0004037151235)
