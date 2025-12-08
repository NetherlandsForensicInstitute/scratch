import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_almost_equal

from conversion.exceptions import NegativeStdScalerException
from image_generation.data_formats import ScanImage
from image_generation.translations import clip_data, grayscale_to_rgba
from utils.array_definitions import ScanMap2DArray


def test_grayscale_to_rgba_converts_nans(scan_image_with_nans: ScanImage):
    rgba = grayscale_to_rgba(scan_image_with_nans.data)
    assert np.array_equal(np.isnan(scan_image_with_nans.data), rgba[..., -1] == 0)


def test_grayscale_to_rgba_has_equal_rgb_channels(scan_image_with_nans: ScanImage):
    rgba = grayscale_to_rgba(scan_image_with_nans.data)
    assert np.array_equal(np.isnan(scan_image_with_nans.data), rgba[..., -1] == 0)
    for channel in range(3):
        assert np.array_equal(
            scan_image_with_nans.data.astype(np.uint8), rgba[..., channel]
        )


def test_grayscale_to_rgba_has_same_size(scan_image_with_nans: ScanImage):
    rgba = grayscale_to_rgba(scan_image_with_nans.data)
    assert rgba.shape[1] == scan_image_with_nans.width
    assert rgba.shape[0] == scan_image_with_nans.height


@pytest.mark.parametrize(
    "given_scan_image",
    [
        pytest.param(
            np.array([[-50.0, 100.0, 300.0], [150.0, 500.0, -20.0]]),
            id="values bigger than 255 and lower than 0",
        ),
        pytest.param(
            np.array([[100.0, -10.0, 200.0], [50.0, 150.0, -5.0]]),
            id="negative values",
        ),
        pytest.param(
            np.array([[100.0, 200.0, 300.0], [50.0, 400.0, 150.0]]),
            id="values bigger than 255",
        ),
    ],
)
def test_grayscale_to_rgba_invalid_values(given_scan_image: ScanMap2DArray) -> None:
    # Arrange
    scan_data = np.array([[-50.0, 100.0, 300.0], [150.0, 500.0, -20.0]])

    # Act & Assert
    with pytest.raises(ValueError, match="values outside \\[0:255\\] range"):
        grayscale_to_rgba(scan_data)


@given(
    std_scaler=st.floats(min_value=0, max_value=100, exclude_min=True),
    data=arrays(
        dtype=float,
        shape=(4, 4),
        elements=st.floats(allow_nan=False, min_value=0.0, max_value=100.0),
    ),
)
def test_clip_data_bounds_match_expected_bounds(
    data: ScanMap2DArray, std_scaler: float
):
    expected_lower = np.mean(data) - np.std(data, ddof=1) * std_scaler
    expected_upper = np.mean(data) + np.std(data, ddof=1) * std_scaler

    clipped, lower, upper = clip_data(data, std_scaler)
    assert np.isclose(lower, expected_lower), f"Lower bound should be {expected_lower}"
    assert np.isclose(upper, expected_upper), f"Upper bound should be {expected_upper}"
    assert clipped.min() >= lower, f"Minimum value should be clipped to {lower}"
    assert clipped.max() <= upper, f"Maximum value should be clipped to {upper}"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(np.array([0, 5, 6.7, 0.12]), id="Array contains no NaNs"),
        pytest.param(
            np.array([0, 5, np.nan, 6.7, 0.12]), id="Array contains single NaN value"
        ),
        pytest.param(
            np.array([0, 5, np.nan, 6.7, 0.12, np.nan, np.nan]),
            id="Array contains multiple NaN values",
        ),
    ],
)
def test_clip_data_ignores_nans(data: ScanMap2DArray):
    std_scaler = 1.0
    expected_lower = -0.45949361789807286
    expected_upper = 6.369493617898073
    _, lower, upper = clip_data(data, std_scaler)
    assert np.isclose(lower, expected_lower), f"Lower bound should be {expected_lower}"
    assert np.isclose(upper, expected_upper), f"Upper bound should be {expected_upper}"


@given(value=st.integers(max_value=100))
def test_no_clipping_when_input_is_constant(value: int):
    data = np.full((4, 4), value)
    std_scaler = 1
    clipped, lower, upper = clip_data(data, std_scaler)

    assert_array_almost_equal(clipped, data)
    assert np.isclose(lower, value)
    assert np.isclose(upper, value)


@given(std_scaler=st.floats(max_value=0))
def test_clip_data_rejects_negative_scalers(
    scan_image_with_nans: ScanImage, std_scaler: float
):
    with pytest.raises(NegativeStdScalerException):
        _ = clip_data(scan_image_with_nans.data, std_scaler)
