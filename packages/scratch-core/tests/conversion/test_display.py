import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from matplotlib.testing.decorators import image_comparison

from conversion.display import clip_data, get_image_for_display, grayscale_to_rgba
from conversion.exceptions import NegativeStdScalerException
from parsers import ScanImage
from ..helper_functions import plot_test_data

PRECISION = 1e-16


@given(std_scaler=st.floats(min_value=0, exclude_min=True))
def test_image_is_clipped_correctly(scan_image_with_nans: ScanImage, std_scaler: float):
    data = scan_image_with_nans.data
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    mean, std = np.nanmean(data), np.nanstd(data, ddof=1) * std_scaler

    clipped, lower, upper = clip_data(data, std_scaler)
    assert np.isclose(lower, mean - std, atol=PRECISION)
    assert np.isclose(upper, mean + std, atol=PRECISION)

    clipped_min, clipped_max = np.nanmin(clipped), np.nanmax(clipped)
    assert lower <= clipped_min
    assert upper >= clipped_max
    assert data_min <= clipped_min
    assert data_max >= clipped_max

    assert np.array_equal(np.isnan(data), np.isnan(clipped))


@given(std_scaler=st.floats(max_value=0))
def test_clip_data_rejects_incorrect_scalers(
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
    display_image = get_image_for_display(scan_image_with_nans)
    plot_test_data(display_image)
