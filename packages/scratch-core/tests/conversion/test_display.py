from conversion.display import clip_data, get_image_for_display
from parsers import ScanImage
import numpy as np
import pytest


@pytest.mark.parametrize("std_scaler", [0.5, 1, 2, 4, 8])
def test_image_is_clipped_correctly(scan_image_with_nans: ScanImage, std_scaler: float):
    data = scan_image_with_nans.data
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    clipped, lower, upper = clip_data(data, std_scaler=std_scaler)
    clipped_data_min, clipped_data_max = np.nanmin(clipped), np.nanmax(clipped)
    assert data_min <= clipped_data_min
    assert lower <= clipped_data_min
    assert data_max >= clipped_data_max
    assert upper >= clipped_data_max
    assert np.array_equal(np.isnan(data), np.isnan(clipped))


def test_get_image_for_display(scan_image_with_nans: ScanImage):
    display_image = get_image_for_display(scan_image_with_nans)
    assert display_image.width == scan_image_with_nans.width
    assert display_image.height == scan_image_with_nans.height
    assert display_image.mode == "RGBA"
    image_data = np.asarray(display_image)
    assert np.array_equal(np.isnan(scan_image_with_nans.data), image_data[..., -1] == 0)
