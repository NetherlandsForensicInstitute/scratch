import numpy as np
import pytest

from image_generation.translations import normalize_2d_array

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
    normalized_image = normalize_2d_array(image, scale_max=max_val, scale_min=min_val)

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
    normalized = normalize_2d_array(image, scale_max=max_value, scale_min=min_val)

    # Assert
    assert np.all(normalized >= min_val)
    assert np.all(normalized <= max_value)
    assert np.allclose(image, normalized, atol=TOLERANCE), (
        "should be the same output as the already normalized input"
    )
