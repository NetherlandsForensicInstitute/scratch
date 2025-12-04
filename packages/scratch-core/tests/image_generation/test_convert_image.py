import numpy as np
from PIL.Image import Image

from image_generation.data_formats import ScanImage


def test_create_image(scan_image_replica: ScanImage) -> None:
    image = scan_image_replica.image

    assert isinstance(image, Image)
    assert image.size == scan_image_replica.data.shape
    assert image.mode == "RGBA"


def test_create_small_image_small_values() -> None:
    # Arrange
    input_data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    input_image = ScanImage(data=input_data, scale_x=1, scale_y=1)
    expected_rgba = np.zeros((*input_data.shape, 4), dtype=np.uint8)
    expected_rgba[..., 0] = input_data
    expected_rgba[..., 1] = input_data
    expected_rgba[..., 2] = input_data
    expected_rgba[..., 3] = 255

    # Act
    output_array = np.array(input_image.image)

    # Assert
    assert np.array_equal(output_array, expected_rgba)
