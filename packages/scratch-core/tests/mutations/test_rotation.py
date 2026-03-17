import numpy as np
import pytest

from container_models.scan_image import ScanImage
from mutations import Rotate


class TestRotate:
    TEST_IMAGE_SIZE = 20

    @pytest.fixture
    def smal_square_scan_image(
        self,
    ) -> ScanImage:
        """Create a small square as a scan_image"""
        x_vals = np.arange(self.TEST_IMAGE_SIZE, dtype=float)
        y_vals = np.arange(self.TEST_IMAGE_SIZE, dtype=float)
        return ScanImage(data=y_vals[:, None] + x_vals[None, :], scale_x=1, scale_y=1)

    @pytest.fixture
    def small_square_image_rotate_left_90(
        self,
    ) -> ScanImage:
        """Rotates the ScanImage 90 degrees to the left."""
        x_vals = np.arange(self.TEST_IMAGE_SIZE, dtype=float)
        y_vals = np.arange(
            self.TEST_IMAGE_SIZE - 1, -1, -1, dtype=float
        )  # inverse y vals to make it look like a left rotation
        return ScanImage(data=y_vals[:, None] + x_vals[None, :], scale_x=1, scale_y=1)

    @pytest.fixture
    def small_sqaure_scan_image_rotate_right_90(
        self,
    ) -> ScanImage:
        """Rotates the ScanImage 90 degrees to the right."""
        x_vals = np.arange(
            self.TEST_IMAGE_SIZE - 1, -1, -1, dtype=float
        )  # inverse x vals to make it look like a right rotation
        y_vals = np.arange(self.TEST_IMAGE_SIZE, dtype=float)
        return ScanImage(data=y_vals[:, None] + x_vals[None, :], scale_x=1, scale_y=1)

    def assert_scan_image_with_expected_scan_image(
        self,
        scan_image: ScanImage,
        expected_scan_image: ScanImage,
        rotated_angle: float,
    ) -> None:
        assert np.allclose(scan_image.data, expected_scan_image.data), (
            f"image should be rotated bij {rotated_angle} degrees."
        )
        assert scan_image.scale_y == expected_scan_image.scale_y, (
            f"scale should be same as original: {expected_scan_image.scale_y}, but got : {scan_image.scale_y}"
        )
        assert scan_image.scale_x == expected_scan_image.scale_x, (
            f"scale should be same as original: {expected_scan_image.scale_x}, but got : {scan_image.scale_x}"
        )

    def test_image_rotates_to_given_angle(
        self,
        smal_square_scan_image: ScanImage,
        small_square_image_rotate_left_90: ScanImage,
    ) -> None:
        # Arrange
        rotation = 90.0
        rotator = Rotate(
            rotation_angle=rotation,
        )
        # Act
        rotated_image = rotator.apply_on_image(scan_image=smal_square_scan_image)
        # Assert
        self.assert_scan_image_with_expected_scan_image(
            scan_image=rotated_image,
            expected_scan_image=small_square_image_rotate_left_90,
            rotated_angle=rotation,
        )

    @pytest.mark.parametrize(
        "rotation",
        [
            pytest.param(0.0, id="Baseline 0 rotation"),
            pytest.param(0.00000001, id="Border of default value e-tolerance"),
            pytest.param(
                -0.00000001, id="Negative border of default value e-tolerance"
            ),
        ],
    )
    def test_image_can_handle_almost_no_rotation(
        self,
        smal_square_scan_image: ScanImage,
        caplog: pytest.LogCaptureFixture,
        rotation: float,
    ) -> None:
        # Arrange
        rotator = Rotate(
            rotation_angle=rotation,
        )
        # Act
        rotated_image = rotator(scan_image=smal_square_scan_image)
        # Assert
        self.assert_scan_image_with_expected_scan_image(
            scan_image=rotated_image,
            expected_scan_image=rotated_image,
            rotated_angle=rotation,
        )
        assert (
            f"No rotation is needed, given rotation angle is close by 0, given angle : {rotation}"
            in caplog.messages
        )
