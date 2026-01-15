import numpy as np
import pytest
from unittest.mock import patch

from container_models.scan_image import ScanImage
from conversion.data_formats import CropType, CropInfo
from conversion.rotate import (
    get_rotation_angle,
    dilate_and_crop_image_and_mask,
    rotate_and_crop_scan_image,
    rotate_and_crop_scan_image_long_version,
    rotate_crop_and_mask_image_by_crop,
)


class TestGetRotationAngle:
    """Test suite for get_rotation_angle function."""

    def test_explicit_rotation_angle_returned(self):
        """Test that explicit rotation angle is returned when non-zero."""
        rotation_angle = 45.0
        result = get_rotation_angle(None, rotation_angle=rotation_angle)
        assert result == 45.0

    def test_zero_rotation_angle_no_crop_info(self):
        """Test that zero is returned when no crop info is provided."""
        result = get_rotation_angle(crop_info=None, rotation_angle=0.0)
        assert result == 0.0

    def test_zero_rotation_angle_empty_crop_info(self):
        """Test that zero is returned when crop info list is empty."""
        result = get_rotation_angle(crop_info=[], rotation_angle=0.0)
        assert result == 0.0

    def test_rotation_from_rectangle_crop_horizontal(self):
        """Test rotation calculation from horizontal rectangle."""
        # Rectangle aligned with axes (0 degrees)
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
        crop_info = [
            CropInfo(
                crop_type=CropType.RECTANGLE,
                data={"corner": corners},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=0.0)
        assert abs(result) < 1e-10

    def test_rotation_from_rectangle_crop_45_degrees(self):
        """Test rotation calculation from 45-degree rotated rectangle."""
        # Rectangle rotated 45 degrees
        angle_rad = np.radians(45)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        corners = np.array(
            [
                [0, 0],
                [100 * cos_a, 100 * sin_a],
                [100 * cos_a - 50 * sin_a, 100 * sin_a + 50 * cos_a],
                [-50 * sin_a, 50 * cos_a],
            ]
        )
        crop_info = [
            CropInfo(
                crop_type=CropType.RECTANGLE,
                data={"corner": corners},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=0.0)
        assert abs(result - 45.0) < 1.0

    def test_rotation_normalization_greater_than_90(self):
        """Test that angles > 90 are normalized to range [-90, 90]."""
        # Create corners that would result in angle > 90
        corners = np.array([[0, 0], [0, 100], [-50, 100], [-50, 0]])
        crop_info = [
            CropInfo(
                crop_type=CropType.RECTANGLE,
                data={"corner": corners},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=0.0)
        assert -90 <= result <= 90

    def test_rotation_normalization_less_than_minus_90(self):
        """Test that angles < -90 are normalized to range [-90, 90]."""
        # Create corners with negative angle
        corners = np.array([[0, 0], [-70.7, -70.7], [-35.3, -106.1], [35.3, -35.3]])
        crop_info = [
            CropInfo(
                crop_type=CropType.RECTANGLE,
                data={"corner": corners},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=0.0)
        assert -90 <= result <= 90

    def test_non_rectangle_crop_type_returns_zero(self):
        """Test that non-RECTANGLE crop types return zero rotation."""
        crop_info = [
            CropInfo(
                crop_type=CropType.CIRCLE,
                data={"center": np.array([3, 4]), "radius": 1.0},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=0.0)
        assert result == 0.0

    def test_explicit_angle_overrides_crop_info(self):
        """Test that explicit rotation angle takes precedence over crop info."""
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
        crop_info = [
            CropInfo(
                crop_type=CropType.RECTANGLE,
                data={"corner": corners},
                is_foreground=True,
            )
        ]
        result = get_rotation_angle(crop_info=crop_info, rotation_angle=30.0)
        assert result == 30.0


class TestDilateAndCropImageAndMask:
    """Test suite for dilate_and_crop_image_and_mask function."""

    def test_no_rotation_no_dilation(self):
        """Test that no dilation occurs when rotation angle is zero."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        mask = np.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = dilate_and_crop_image_and_mask(
            scan_image, mask, rotation_angle=0.0
        )

        assert isinstance(result_image, ScanImage)
        assert result_image.scale_x == 1.0
        assert result_image.scale_y == 1.0
        assert result_image.data.shape[0] == 2  # Cropped to mask bounds
        assert result_image.data.shape[1] == 2

    def test_with_rotation_applies_dilation(self):
        """Test that dilation is applied when rotation angle is non-zero."""
        data = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True  # Small region in center
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = dilate_and_crop_image_and_mask(
            scan_image, mask, rotation_angle=45.0
        )

        # After 3 iterations of dilation, the mask should be larger
        assert result_mask.sum() > mask.sum()

    def test_preserves_scale_values(self):
        """Test that scale_x and scale_y are preserved."""
        data = np.ones((5, 5), dtype=float)
        mask = np.ones((5, 5), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=2.5, scale_y=3.7)

        result_image, result_mask = dilate_and_crop_image_and_mask(
            scan_image, mask, rotation_angle=0.0
        )

        assert result_image.scale_x == 2.5
        assert result_image.scale_y == 3.7

    def test_output_types(self):
        """Test that output types are correct."""
        data = np.ones((5, 5), dtype=float)
        mask = np.ones((5, 5), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = dilate_and_crop_image_and_mask(
            scan_image, mask, rotation_angle=0.0
        )

        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)
        assert result_mask.dtype == bool


class TestRotateAndCropScanImage:
    """Test suite for rotate_and_crop_scan_image function."""

    def test_zero_rotation_returns_unchanged(self):
        """Test that zero rotation returns original data."""
        data = np.array([[1, 2], [3, 4]], dtype=float)
        mask = np.array([[True, True], [True, False]])
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_and_crop_scan_image(
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

        result_image, result_mask = rotate_and_crop_scan_image(
            scan_image, mask, rotation_angle=45
        )

        # After rotation, shape should change
        assert result_image.data.shape != data.shape

    def test_rotation_handles_nan_values(self):
        """Test that rotation properly handles NaN values in data."""
        data = np.ones((10, 10), dtype=float)
        data[0:2, 0:2] = np.nan
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_and_crop_scan_image(
            scan_image, mask, rotation_angle=30
        )

        # Should not raise error and should handle NaNs
        assert isinstance(result_image, ScanImage)

    def test_empty_rotated_mask_raises_error(self):
        """Test that empty rotated mask raises ValueError."""
        data = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)  # Empty mask
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        with pytest.raises(ValueError, match="Rotated mask is empty"):
            rotate_and_crop_scan_image(scan_image, mask, rotation_angle=45)

    def test_preserves_scale_after_rotation(self):
        """Test that scale values are preserved after rotation."""
        data = np.ones((10, 10), dtype=float)
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=2.0, scale_y=3.0)

        result_image, result_mask = rotate_and_crop_scan_image(
            scan_image, mask, rotation_angle=15
        )

        assert result_image.scale_x == 2.0
        assert result_image.scale_y == 3.0

    def test_cropping_applies_margin(self):
        """Test that cropping applies the specified margin."""
        data = np.ones((20, 20), dtype=float)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_and_crop_scan_image(
            scan_image, mask, rotation_angle=10
        )

        # After applying margin, the cropped region should be smaller
        assert result_image.data.shape[0] < 20
        assert result_image.data.shape[1] < 20


class TestRotateAndCropScanImageLongVersion:
    """Test suite for rotate_and_crop_scan_image_long_version function."""

    def test_rotation_with_nans(self):
        """Test rotation handling when data contains NaN values."""
        data = np.ones((20, 20), dtype=float)
        data[0:5, 0:5] = np.nan
        mask = np.ones((20, 20), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_obj, result_mask = rotate_and_crop_scan_image_long_version(
            scan_image, mask, rotation_angle=30
        )

        assert hasattr(result_obj, "depth_data")
        assert isinstance(result_mask, np.ndarray)

    def test_rotation_without_nans(self):
        """Test rotation handling when data has no NaN values."""
        data = np.ones((20, 20), dtype=float)
        mask = np.ones((20, 20), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_obj, result_mask = rotate_and_crop_scan_image_long_version(
            scan_image, mask, rotation_angle=45
        )

        assert hasattr(result_obj, "depth_data")
        assert isinstance(result_mask, np.ndarray)

    def test_empty_rotated_mask_raises_error(self):
        """Test that empty rotated mask raises ValueError."""
        data = np.ones((10, 10), dtype=float)
        mask = np.zeros((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        with pytest.raises(ValueError, match="Rotated mask is empty"):
            rotate_and_crop_scan_image_long_version(scan_image, mask, rotation_angle=45)

    def test_mask_expansion_when_needed(self):
        """Test that mask is expanded when rotated mask is larger than data."""
        data = np.ones((10, 10), dtype=float)
        data[2:8, 2:8] = np.nan
        mask = np.zeros((15, 15), dtype=bool)
        mask[2:13, 2:13] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_obj, result_mask = rotate_and_crop_scan_image_long_version(
            scan_image, mask, rotation_angle=45
        )

        # Should not raise error
        assert hasattr(result_obj, "depth_data")

    def test_dilation_steps_applied_to_cropping(self):
        """Test that dilation steps are applied when cropping."""
        data = np.ones((30, 30), dtype=float)
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_obj, result_mask = rotate_and_crop_scan_image_long_version(
            scan_image, mask, rotation_angle=15
        )

        # Output should be cropped (smaller than input)
        assert result_obj.depth_data.shape[0] < mask.shape[0]
        assert result_obj.depth_data.shape[1] < mask.shape[1]


class TestRotateCropImageFullFlow:
    """Test suite for rotate_crop_image_full_flow function."""

    def test_full_flow_with_zero_rotation(self):
        """Test full flow with zero rotation angle."""
        data = np.ones((10, 10), dtype=float)
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_crop_and_mask_image_by_crop(
            scan_image, mask, rotation_angle=0.0, crop_info=None
        )

        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)

    @patch("conversion.remove_needles")
    def test_full_flow_with_rotation_from_crop_info(self, mock_remove_needles):
        """Test full flow with rotation angle computed from crop info."""
        data = np.ones((20, 20), dtype=float)
        mask = np.ones((20, 20), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        # Create crop info with 45-degree rotation
        angle_rad = np.radians(45)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        corners = np.array(
            [
                [0, 0],
                [100 * cos_a, 100 * sin_a],
                [100 * cos_a - 50 * sin_a, 100 * sin_a + 50 * cos_a],
                [-50 * sin_a, 50 * cos_a],
            ]
        )
        crop_info = [CropInfo(crop_type=CropType.RECTANGLE, data={"corner": corners})]

        # Mock remove_needles to return a scan_image
        mock_remove_needles.return_value = scan_image

        result_image, result_mask = rotate_crop_and_mask_image_by_crop(
            scan_image, mask, rotation_angle=0.0, crop_info=crop_info
        )

        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)

    @patch("conversion.remove_needles")
    def test_full_flow_calls_remove_needles_with_correct_params(
        self, mock_remove_needles
    ):
        """Test that remove_needles is called with correct parameters."""
        data = np.ones((10, 10), dtype=float)
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        mock_remove_needles.return_value = scan_image

        rotate_crop_and_mask_image_by_crop(
            scan_image, mask, rotation_angle=0.0, times_median=20.0
        )

        # Check that remove_needles was called with times_median=20.0
        call_args = mock_remove_needles.call_args
        assert call_args[0][2] == 20.0  # times_median parameter

    @patch("conversion.remove_needles")
    def test_full_flow_preserves_scale(self, mock_remove_needles):
        """Test that scale values are preserved through the full flow."""
        data = np.ones((10, 10), dtype=float)
        mask = np.ones((10, 10), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=2.5, scale_y=3.5)

        mock_remove_needles.return_value = scan_image

        result_image, result_mask = rotate_crop_and_mask_image_by_crop(
            scan_image, mask, rotation_angle=0.0
        )

        assert result_image.scale_x == 2.5
        assert result_image.scale_y == 3.5

    @patch("conversion.remove_needles")
    def test_full_flow_with_explicit_rotation_angle(self, mock_remove_needles):
        """Test full flow with explicit rotation angle."""
        data = np.ones((20, 20), dtype=float)
        mask = np.ones((20, 20), dtype=bool)
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        mock_remove_needles.return_value = scan_image

        result_image, result_mask = rotate_crop_and_mask_image_by_crop(
            scan_image, mask, rotation_angle=30.0, crop_info=None
        )

        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_complete_workflow_no_rotation(self):
        """Test complete workflow without rotation."""
        data = np.random.rand(50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        # Test each step individually
        rotation_angle = get_rotation_angle(crop_info=None, rotation_angle=0.0)
        assert rotation_angle == 0.0

        result_image, result_mask = dilate_and_crop_image_and_mask(
            scan_image, mask, rotation_angle
        )
        assert isinstance(result_image, ScanImage)

    def test_complete_workflow_with_rotation(self):
        """Test complete workflow with rotation."""
        data = np.random.rand(50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 15:35] = True
        scan_image = ScanImage(data=data, scale_x=1.0, scale_y=1.0)

        result_image, result_mask = rotate_and_crop_scan_image(
            scan_image, mask, rotation_angle=15
        )

        assert isinstance(result_image, ScanImage)
        assert isinstance(result_mask, np.ndarray)
        assert result_image.scale_x == 1.0
        assert result_image.scale_y == 1.0
