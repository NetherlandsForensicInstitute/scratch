import numpy as np

import pytest

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.rotate import (
    get_rotation_angle,
    crop_image_and_mask_to_mask,
    rotate_mask_and_scan_image,
)


class TestGetRotationAngle:
    """Test suite for get_rotation_angle function."""

    @pytest.fixture
    def rectangle_15deg(self) -> BoundingBox:
        return np.array(
            [
                [53.69130745, 16.00419511],
                [169.60240661, 47.06248052],
                [146.30869255, 133.99580489],
                [30.39759339, 102.93751948],
            ]
        )

    @pytest.fixture
    def rectangle_0deg(self) -> BoundingBox:
        return np.array(
            [
                [30.0, 23.0],
                [169.0, 23.0],
                [169.0, 126.0],
                [30.0, 126.0],
            ]
        )

    def test_rotation_from_rectangle_crop_0_degrees(self, rectangle_0deg):
        """Test rotation calculation from horizontal rectangle."""
        result = get_rotation_angle(bounding_box=rectangle_0deg)
        assert result == 0

    def test_rotation_from_rectangle_crop_15_degrees(self, rectangle_15deg):
        """Test rotation calculation from 15-degree rotated rectangle."""
        result = get_rotation_angle(bounding_box=rectangle_15deg)
        assert result == pytest.approx(15)

    def test_rotation_normalization(self):
        """Test that angles > 90 are normalized to range [-90, 90]."""
        # Create corners that would result in angle > 90
        rectangle = np.array([[0, 0], [0, 100], [-50, 100], [-50, 0]])
        result = get_rotation_angle(bounding_box=rectangle)
        assert -90 <= result <= 90


class TestCropImageAndMaskToMask:
    """Test suite for crop_image_and_mask_to_mask function."""

    @pytest.fixture
    def scan_image(self) -> ScanImage:
        return ScanImage(data=np.ones((10, 10), dtype=float), scale_x=1.0, scale_y=1.0)

    @pytest.fixture
    def mask(self) -> MaskArray:
        mask = np.zeros((10, 10), dtype=float)
        mask[2:8, 2:8] = True
        return mask

    @pytest.mark.parametrize(
        "margin, output_shape",
        [
            pytest.param(0, (6, 6), id="Normal bounding box, no margin"),
            pytest.param(0, (6, 6), id="Normal bounding box, no marginx"),
            pytest.param(1, (4, 4), id="Decrease bounding box"),
            pytest.param(-1, (8, 8), id="Increase bounding box"),
            pytest.param(-5, (10, 10), id="Margin bounding box out of bounds"),
        ],
    )
    def test_crop_image_and_mask_to_mask(
        self,
        scan_image: ScanImage,
        mask: MaskArray,
        margin: int,
        output_shape: tuple[int, int],
    ):
        """Test cropping with different margins."""
        result_image, result_mask = crop_image_and_mask_to_mask(
            scan_image, mask, margin=margin
        )

        assert isinstance(result_image, ScanImage)
        assert result_image.scale_x == scan_image.scale_x
        assert result_image.scale_y == scan_image.scale_y

        assert result_image.data.shape == output_shape
        assert result_mask.shape == output_shape


class TestRotateMaskAndScanImage:
    """Test suite for rotate_mask_and_scan_image function."""

    def test_zero_rotation_returns_unchanged(self):
        """Test that zero rotation returns original data."""
        data = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[True, True], [True, False]])
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_mask_and_scan_image(
            scan_image, mask, rotation_angle=0
        )

        np.testing.assert_array_equal(result_image.data, data)
        np.testing.assert_array_equal(result_mask, mask)

    def test_non_zero_rotation_applies_rotation(self):
        """Test that non-zero rotation angle rotates the data."""
        data = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_mask_and_scan_image(
            scan_image, mask, rotation_angle=45
        )

        # After rotation, shape should change due to reshape=True
        assert result_image.data.shape != data.shape or not np.array_equal(
            result_image.data, data
        )

    def test_rotation_handles_nan_values(self):
        """Test that rotation properly handles NaN values in data."""
        data = np.ones((10, 10), dtype=float)
        data[0:2, 0:2] = np.nan
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_mask_and_scan_image(
            scan_image, mask, rotation_angle=30
        )

        # Should not raise error and should handle NaNs
        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)

    def test_preserves_scale_after_rotation(self):
        """Test that scale values are preserved after rotation."""
        data = np.ones((10, 10), dtype=float)
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=2.0, scale_y=3.0)

        result_image, result_mask = rotate_mask_and_scan_image(
            scan_image, mask, rotation_angle=15
        )

        assert result_image.scale_x == 2.0
        assert result_image.scale_y == 3.0
