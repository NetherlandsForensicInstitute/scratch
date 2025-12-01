import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from matplotlib.testing.decorators import image_comparison
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from conversion.display import clip_data, get_array_for_display, grayscale_to_rgba
from conversion.exceptions import NegativeStdScalerException
from parsers import ScanImage
from ..helper_functions import plot_test_data  # type: ignore


@given(
    std_scaler=st.floats(min_value=0, exclude_min=True),
    data=arrays(
        dtype=float,
        shape=(4, 4),
        elements=st.floats(allow_nan=False, min_value=0.0, max_value=100.0),
    ),
)
def test_clip_data_basic(data: NDArray, std_scaler: float):
    expected_lower = np.mean(data) - np.std(data, ddof=1) * std_scaler
    expected_upper = np.mean(data) + np.std(data, ddof=1) * std_scaler

    _, lower, upper = clip_data(data, std_scaler)

    # Check if the lower and upper bounds are correct
    assert np.isclose(lower, expected_lower)
    assert np.isclose(upper, expected_upper)


def test_clip_data_with_nan():
    data = np.array([1, 2, 3, np.nan, 5])
    std_scaler = 1
    clipped, lower, upper = clip_data(data, std_scaler)

    # NaN values should be ignored, and other values clipped accordingly
    assert np.isnan(data[3])  # Ensure NaN is present in original data
    assert np.isclose(
        lower,
        np.mean(data[~np.isnan(data)])
        - np.std(data[~np.isnan(data)], ddof=1) * std_scaler,
    )
    assert np.isclose(
        upper,
        np.mean(data[~np.isnan(data)])
        + np.std(data[~np.isnan(data)], ddof=1) * std_scaler,
    )


@given(val=st.integers(max_value=100))
def test_clip_data_identical_values(val):
    data = np.full((4, 4), val)
    std_scaler = 1
    clipped, lower, upper = clip_data(data, std_scaler)

    assert_array_almost_equal(clipped, data)  # All values should remain the same
    assert np.isclose(lower, val)
    assert np.isclose(upper, val)


@given(std_scaler=st.floats(max_value=0))
def test_clip_data_rejects_negative_scalers(
    scan_image_with_nans: ScanImage, std_scaler: float
):
    with pytest.raises(NegativeStdScalerException):
        _ = clip_data(scan_image_with_nans.data, std_scaler)


def test_grayscale_to_rgba(scan_image_with_nans: ScanImage):
    rgba = grayscale_to_rgba(scan_image_with_nans.data)
    assert rgba.shape[1] == scan_image_with_nans.width
    assert rgba.shape[0] == scan_image_with_nans.height
    assert np.array_equal(np.isnan(scan_image_with_nans.data), rgba[..., -1] == 0)
    for channel in range(3):
        assert np.array_equal(
            scan_image_with_nans.data.astype(np.uint8), rgba[..., channel]
        )


@pytest.mark.integration
@image_comparison(baseline_images=["preview_image"], extensions=["png"])
def test_get_image_for_display_matches_baseline_image(
    scan_image_with_nans: ScanImage,
):
    display_image = get_array_for_display(scan_image_with_nans)
    plot_test_data(display_image)
