from conversion.exceptions import ConversionError
import numpy as np
from PIL.Image import Image

import pytest
from image_generation.data_formats import ScanImage


def test_create_image(scan_image_replica: ScanImage) -> None:
    image = scan_image_replica.image()

    assert isinstance(image, Image)
    assert image.size == scan_image_replica.data.shape
    assert image.mode == "RGBA"


def test_create_small_image_small_values() -> None:
    # Arrange
    input_data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    input_image = ScanImage(data=input_data, scale_x=1, scale_y=1)
    expected_rgba = np.stack(
        ([input_data] * 3 + [np.full_like(input_data, 255)]),
        axis=-1,
    ).astype(np.uint8)

    # Act
    output_array = np.array(input_image.image())

    # Assert
    assert np.array_equal(output_array, expected_rgba)


def test_create_wrong_values() -> None:
    # Arrange
    input_data = np.array([[-50.0, 100.0, 300.0], [150.0, 500.0, -20.0]])
    input_image = ScanImage(data=input_data, scale_x=1, scale_y=1)

    # Act
    with pytest.raises(ConversionError):
        input_image.image()
