import numpy as np
import pytest

from conversion.leveling.solver import compute_root_mean_square


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0, -3.15, 40.123, -80, 100])
def test_rms_is_constant_for_constant_input(value: float):
    result = compute_root_mean_square(np.array([value] * 100))
    assert np.isclose(result, abs(value))


@pytest.mark.parametrize("value", list(range(-10, 10)))
def test_rms_is_non_negative(value: float):
    result = compute_root_mean_square(np.array([value] * 100))
    assert result > 0 or np.isclose(result, 0.0)


def test_rms_can_handle_nans():
    array_with_nans = np.array(
        [0, 1, 0.15, 2, np.nan, 3, 4, np.nan, -5], dtype=np.float64
    )
    array_without_nans = array_with_nans[~np.isnan(array_with_nans)]

    result_with_nans = compute_root_mean_square(array_with_nans)
    result_without_nans = compute_root_mean_square(array_without_nans)
    assert np.isclose(result_with_nans, result_without_nans)
    assert np.isclose(result_with_nans, 2.8036328473709147)
