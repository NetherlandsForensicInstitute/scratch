from container_models.scan_image import ScanImage
from mutations.spatial import Crop
import pytest
import numpy as np


class TestCropImage:
    @pytest.fixture
    def scan_image(self) -> ScanImage:
        data = np.arange(16, dtype=np.float32).reshape((4, 4))
        return ScanImage(data=data.copy(), scale_x=1, scale_y=1)

    def test_crop_applies_bounding_box(self, scan_image: ScanImage):
        # Arrange
        mask = np.ones(scan_image.data.shape, dtype=bool)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
        crop = Crop(crop=mask)

        # Act
        result = crop(scan_image).unwrap()

        # Assert
        removed_border = np.array([[5, 6], [9, 10]], dtype=scan_image.data.dtype)
        assert result.data.shape == (2, 2)
        np.testing.assert_array_equal(result.data, removed_border)
        assert result.scale_x == scan_image.scale_x, (
            "scale should be the same (unchanged)"
        )
        assert result.scale_y == scan_image.scale_y, (
            "scale should be the same (unchanged)"
        )

    def test_crop_skipped_when_predicate_true(self, scan_image: ScanImage):
        # Arrange
        crop = Crop(crop=(np.zeros(scan_image.data.shape, dtype=np.bool)))
        # Act
        result = crop(scan_image).unwrap()
        # Assert
        assert result is scan_image
        np.testing.assert_array_equal(result.data, scan_image.data)
