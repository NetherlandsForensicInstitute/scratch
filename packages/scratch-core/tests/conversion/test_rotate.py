import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import CropInfo
from conversion.rotate import (
    get_rotation_angle,
    rotate_scan_image,
)


class TestGetRotationAngle:
    def test_get_rotation_angle_straight_rectangle(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_single_rectangle: list[CropInfo],
    ):
        """Assert that when a non-skewed rectangle is first in the crop_info list, the rotation angle is 0.0."""
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate, crop_info=crop_info_single_rectangle
        )
        assert rotation_angle == 0.0

    def test_get_rotation_angle_given_rotation_angle(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_multiple_shapes_rectangle_first: list[CropInfo],
    ):
        """Assert that when a rotation angle is given by the user, that rotation angle is returned, even if
        the first shape in crop_info is a skewed rectangle."""
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            rotation_angle=5.0,
            crop_info=crop_info_multiple_shapes_rectangle_first,
        )
        assert rotation_angle == 5.0

    def test_get_rotation_angle_multi_shape_rectangle_first(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_multiple_shapes_rectangle_first: list[CropInfo],
    ):
        """Assert that when a (skewed) rectangle is first in the crop_info list, the rotation angle is calculated."""
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            crop_info=crop_info_multiple_shapes_rectangle_first,
        )
        # TODO: verify this
        assert rotation_angle == pytest.approx(57.13, rel=1e-3)

    def test_get_rotation_angle_multi_shape_rectangle_last(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_multiple_shapes_rectangle_not_first: list[CropInfo],
    ):
        """
        Assert that when the (skewed) rectangle is not first in the crop_info list, no rotation angle is calculated.
        """
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            crop_info=crop_info_multiple_shapes_rectangle_not_first,
        )
        assert rotation_angle == 0.0


class TestRotateScanImage:
    def test_rotate_scan_image(self, scan_image_before_crop_and_rotate: ScanImage):
        _ = rotate_scan_image(scan_image_before_crop_and_rotate, rotation_angle=50)
        # TODO: determine ground truth, assert things
