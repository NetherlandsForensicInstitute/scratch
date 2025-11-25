from conversion.display import clip_data, get_image_for_display
from parsers import ScanImage
import numpy as np
import pytest

PRECISION = 1e-16


@pytest.mark.parametrize("std_scaler", [0.5, 1, 2, 4, 8])
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


def test_get_image_for_display(scan_image_with_nans: ScanImage):
    display_image = get_image_for_display(scan_image_with_nans)
    assert display_image.width == scan_image_with_nans.width
    assert display_image.height == scan_image_with_nans.height
    assert display_image.mode == "RGBA"

    image_data = np.asarray(display_image)
    assert np.array_equal(np.isnan(scan_image_with_nans.data), image_data[..., -1] == 0)
