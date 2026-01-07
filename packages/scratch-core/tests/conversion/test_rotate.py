import pytest

from container_models.scan_image import ScanImage
from conversion.data_formats import CropInfo
from conversion.rotate import get_rotation_angle


class TestGetRotationAngle:
    def test_get_rotation_angle_straight_rectangle(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_single_rectangle: list[CropInfo],
    ):
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate, crop_info=crop_info_single_rectangle
        )
        assert rotation_angle == 0.0

    def test_get_rotation_angle_given_rotation_angle(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_single_rectangle: list[CropInfo],
    ):
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            rotation_angle=5.0,
            crop_info=crop_info_single_rectangle,
        )
        assert rotation_angle == 5.0

    def test_get_rotation_angle_multi_shape_rectangle_first(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_multiple_shapes_rectangle_first: list[CropInfo],
    ):
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            crop_info=crop_info_multiple_shapes_rectangle_first,
        )
        assert rotation_angle == pytest.approx(57.13, rel=1e-3)

    def test_get_rotation_angle_multi_shape_rectangle_last(
        self,
        scan_image_before_crop_and_rotate: ScanImage,
        crop_info_multiple_shapes_rectangle_not_first: list[CropInfo],
    ):
        rotation_angle = get_rotation_angle(
            scan_image_before_crop_and_rotate,
            crop_info=crop_info_multiple_shapes_rectangle_not_first,
        )
        assert rotation_angle == 0.0
