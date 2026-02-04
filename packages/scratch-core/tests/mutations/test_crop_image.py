import re
from container_models.image import ImageContainer, MetaData
from exceptions import ImageShapeMismatchError
from mutations.spatial import CropToMask
import pytest
import numpy as np


class TestCropImage:
    @pytest.fixture
    def scan_image(self, flat_scale: MetaData) -> ImageContainer:
        return ImageContainer(
            data=np.arange(16, dtype=np.float32).reshape((4, 4)), metadata=flat_scale
        )

    def test_crop_applies_bounding_box(self, scan_image: ImageContainer):
        # Arrange
        mask = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        )
        crop = CropToMask(mask=mask)
        # Act
        result = crop(scan_image).unwrap()

        # Assert
        removed_border = np.array([[5, 6], [9, 10]], dtype=scan_image.data.dtype)
        assert result.data.shape == (2, 2)
        np.testing.assert_array_equal(result.data, removed_border)
        assert result.metadata.scale == scan_image.metadata.scale, (
            "scale should be the same (unchanged)"
        )

    def test_crop_skipped_when_predicate_true(self, scan_image: ImageContainer):
        # Arrange
        crop = CropToMask(mask=(np.ones(scan_image.data.shape, dtype=np.bool)))
        # Act
        result = crop(scan_image).unwrap()
        # Assert
        assert result is scan_image
        np.testing.assert_array_equal(result.data, scan_image.data)

    def test_crop_cropped_all(self, scan_image: ImageContainer):
        # Act / Assert
        with pytest.raises(
            ValueError,
            match=re.escape("Can't crop to a mask where there are only 0/False"),
        ):
            _ = CropToMask(mask=(np.zeros(scan_image.data.shape, dtype=np.bool)))

    def test_image_and_crop_not_equal_in_size(self, scan_image: ImageContainer):
        # Arrange
        offset_size = 1
        cropping_mutator = CropToMask(
            mask=(
                np.ones(
                    (scan_image.width - offset_size, scan_image.height + offset_size),
                    dtype=np.bool,
                )
            )
        )
        expected_error_message = f"image shape: {scan_image.data.shape} and crop shape: {cropping_mutator.mask.shape} are not equal"
        # Act/ Assert
        with pytest.raises(
            ImageShapeMismatchError, match=re.escape(expected_error_message)
        ):
            _ = cropping_mutator.apply_on_image(scan_image)
