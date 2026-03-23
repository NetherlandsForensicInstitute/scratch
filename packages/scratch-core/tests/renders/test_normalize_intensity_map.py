import numpy as np
import pytest

from container_models.models import NormalizationBounds
from container_models.scan_image import ScanImage, _normalize_2d_array

from ..helper_function import assert_nan_mask_preserved

TEST_IMAGE_WIDTH = 10
TEST_IMAGE_HEIGHT = 12
TOLERANCE = 1e-5


@pytest.mark.parametrize(
    "start_value, slope",
    [
        pytest.param(10, 100.0, id="test bigger numbers are reduced"),
        pytest.param(-200, 10.0, id="test negative numbers are upped"),
        pytest.param(100, 0.01, id="small slope is streched over the range"),
    ],
)
def test_bigger_numbers(start_value: int, slope: float) -> None:
    # Arrange
    row = (start_value + slope * np.arange(TEST_IMAGE_WIDTH)).astype(np.float64)
    image = np.tile(row, (TEST_IMAGE_HEIGHT, 1)).astype(np.float64)
    max_val = 255
    min_val = 20
    # Act
    normalized_image = _normalize_2d_array(
        image, normalization_bounds=NormalizationBounds(high=max_val, low=min_val)
    )

    # Assert
    assert normalized_image.max() <= max_val
    assert normalized_image.min() >= min_val
    assert normalized_image[0, 0] == normalized_image.min()
    assert normalized_image[9, 9] == normalized_image.max()


def test_already_normalized_image() -> None:
    # Arrange
    max_value = 255
    min_val = 20
    image = np.linspace(
        min_val, max_value, num=TEST_IMAGE_WIDTH * TEST_IMAGE_HEIGHT
    ).reshape(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)

    # Act
    normalized = _normalize_2d_array(
        array_to_normalize=image,
        normalization_bounds=NormalizationBounds(high=max_value, low=min_val),
    )

    # Assert
    assert np.all(normalized >= min_val)
    assert np.all(normalized <= max_value)
    assert np.allclose(image, normalized, atol=TOLERANCE), (
        "should be the same output as the already normalized input"
    )


def test_nan_mask_preserved(image_with_nan_background: ScanImage):
    result = _normalize_2d_array(
        image_with_nan_background.data,
        normalization_bounds=NormalizationBounds(high=255, low=25),
    )
    assert_nan_mask_preserved(image_with_nan_background.data, result)


def test_constant_valid_region_does_not_produce_nan(
    image_with_nan_background: ScanImage,
):
    """When all valid pixels have the same value, normalization should not
    introduce NaN via division by zero."""
    constant_data = image_with_nan_background.data.copy()
    constant_data[~np.isnan(constant_data)] = 42.0

    result = _normalize_2d_array(
        constant_data, normalization_bounds=NormalizationBounds(high=255, low=25)
    )

    valid_mask = ~np.isnan(constant_data)
    assert np.all(np.isfinite(result[valid_mask])), (
        "Constant valid region should produce finite values, not NaN"
    )
    assert_nan_mask_preserved(constant_data, result)
