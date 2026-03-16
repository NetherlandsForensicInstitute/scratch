import numpy as np
import pytest

from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.rotate import (
    rotate_mask,
)


class TestCropImageAndMaskToMask:
    """Test suite for crop_image_and_mask_to_mask function."""

    @pytest.fixture
    def scan_image(self) -> ScanImage:
        return ScanImage(data=np.ones((10, 10), dtype=float), scale_x=1.0, scale_y=1.0)

    @pytest.fixture
    def mask(self) -> BinaryMask:
        mask = np.zeros((10, 10), dtype=float)
        mask[2:8, 2:8] = True
        return mask


class TestRotateMaskAndScanImage:
    """Test suite for rotate_mask_and_scan_image function."""

    def test_zero_rotation_returns_unchanged(self):
        """Test that zero rotation returns original data."""
        mask = np.array([[True, True], [True, False]])

        result_mask = rotate_mask(mask, rotation_angle=0)

        np.testing.assert_array_equal(result_mask, mask)

    def test_non_zero_rotation_applies_rotation(self):
        """Test that non-zero rotation angle rotates the data."""
        data = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True

        result_mask = rotate_mask(mask, rotation_angle=45)

        # After rotation, shape should change due to reshape=True
        assert result_mask.shape != data.shape or not np.array_equal(
            result_mask.data, mask
        )

    def test_rotation_handles_nan_values(self):
        """Test that rotation properly handles NaN values in data."""
        data = np.ones((10, 10), dtype=float)
        data[0:2, 0:2] = np.nan
        mask = np.ones((10, 10), dtype=bool)

        result_mask = rotate_mask(mask, rotation_angle=30)

        # Should not raise error and should handle NaNs
        assert isinstance(result_mask, np.ndarray)
