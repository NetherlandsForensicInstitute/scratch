import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from conversion.display import clip_data, get_array_for_display, grayscale_to_rgba
from conversion.exceptions import NegativeStdScalerException
from parsers import ScanImage

from ..constants import BASELINE_IMAGES_DIR  # type: ignore


@given(
    std_scaler=st.floats(min_value=0, max_value=100, exclude_min=True),
    data=arrays(
        dtype=float,
        shape=(4, 4),
        elements=st.floats(allow_nan=False, min_value=0.0, max_value=100.0),
    ),
)
def test_clip_data_bounds_match_expected_bounds(data: NDArray, std_scaler: float):
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
def test_clip_data_ignores_nans(data: NDArray):
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

    assert_array_almost_equal(clipped, data), "All values should remain constant"
    assert np.isclose(lower, value)
    assert np.isclose(upper, value)


@given(std_scaler=st.floats(max_value=0))
def test_clip_data_rejects_negative_scalers(
    scan_image_with_nans: ScanImage, std_scaler: float
):
    with pytest.raises(NegativeStdScalerException):
        _ = clip_data(scan_image_with_nans.data, std_scaler)


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


@pytest.mark.integration
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage,
):
    verified = np.load(BASELINE_IMAGES_DIR / "display_array.npy")
    display_image = get_array_for_display(scan_image_with_nans)
    assert_array_almost_equal(display_image, verified)
