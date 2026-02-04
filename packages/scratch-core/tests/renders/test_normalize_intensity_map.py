import numpy as np
import pytest
from scipy.constants import micro

from renders.normalizations import normalize_2d_array, MIN_SCALE, MAX_SCALE


TEST_IMAGE_WIDTH = 10
TEST_IMAGE_HEIGHT = 12


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
    # Act
    normalized_image = normalize_2d_array(image)

    # Assert
    assert normalized_image.max() <= MAX_SCALE
    assert normalized_image.min() >= MIN_SCALE
    assert normalized_image[0, 0] == normalized_image.min()
    assert normalized_image[9, 9] == normalized_image.max()


def test_already_normalized_image() -> None:
    # Arrange
    image = np.linspace(
        MIN_SCALE, MAX_SCALE, num=TEST_IMAGE_WIDTH * TEST_IMAGE_HEIGHT
    ).reshape(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)

    # Act
    normalized = normalize_2d_array(image)

    # Assert
    assert np.all(normalized >= MIN_SCALE)
    assert np.all(normalized <= MAX_SCALE)
    assert np.allclose(image, normalized, atol=micro), (
        "should be the same output as the already normalized input"
    )
