import numpy as np
from scipy.ndimage import binary_dilation, rotate

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import remove_needles
from conversion.data_formats import CropType, CropInfo
from conversion.mask import crop_to_mask, mask_2d_array


def get_rotation_angle(
    scan_image: ScanImage,
    rotation_angle: float = 0.0,
    crop_info: list[CropInfo] = None,
) -> float:
    if (
        rotation_angle == 0.0
        and crop_info
        and crop_info[0].crop_type == CropType.RECTANGLE
    ):
        points = crop_info[0].data["corner"]

        # Make sure the points are ordered counter clockwise in the Java reference,
        # thus first coordinate right, second up
        mean_xy = np.mean(points, axis=0)  # coordinates center of rectangle
        a = np.degrees(
            np.arctan2(points[:, 1] - mean_xy[1], points[:, 0] - mean_xy[0])
        )  # angular position corners wrt center
        order = np.argsort(a)
        points_ordered = points[order, :]

        index = None
        for count in range(4):
            if np.array_equal(points_ordered[count, :], points[0, :]):
                index = count
                break

        if index > 0:
            points = np.vstack([points_ordered[index:, :], points_ordered[:index, :]])
        else:
            points = points_ordered

        # Convert Java coordinates to Matlab coordinates by y-flipping and adding 1
        # NOTE: First dimension is the IMAGE X-AXIS
        points[:, 1] = (scan_image.width - 1) - points[:, 1]
        points = points + 1

        P1 = points[0, :]
        P2 = points[1, :]
        P3 = points[2, :]
        P4 = points[3, :]

        # Determine the rotation angle
        dx = np.array([P2[1] - P1[1], P3[1] - P2[1], P3[1] - P4[1], P4[1] - P1[1]])
        dy = np.array([P2[0] - P1[0], P3[0] - P2[0], P3[0] - P4[0], P4[0] - P1[0]])

        # The angle is a rotation counter clockwise of rotation_angle degrees in
        # the Java reference frame. This corresponds to a rotation clockwise when
        # flipped in the y-dimension.
        rotation_angles = np.degrees(np.arctan2(dx, dy))

        # A rectangle that was not rotated should have resulted in [0, -90, 0, -90]
        # The angle over which each side of the rectangle is rotated is
        rotation_angles = rotation_angles - np.array([0, -90, 0, -90])

        # We know that all 4 rotation angles should be almost similar, but due
        # to the fact that atan2d jumps from -180 to 180 degrees, some strange results might have occurred.
        # Therefore the following checks are performed...
        while rotation_angles[1] - rotation_angles[0] > 180:
            rotation_angles[1] = rotation_angles[1] - 360
        while rotation_angles[1] - rotation_angles[0] < -180:
            rotation_angles[1] = rotation_angles[1] + 360
        while rotation_angles[2] - rotation_angles[0] > 180:
            rotation_angles[2] = rotation_angles[2] - 360
        while rotation_angles[2] - rotation_angles[0] < -180:
            rotation_angles[2] = rotation_angles[2] + 360
        while rotation_angles[3] - rotation_angles[0] > 180:
            rotation_angles[3] = rotation_angles[3] - 360
        while rotation_angles[3] - rotation_angles[0] < -180:
            rotation_angles[3] = rotation_angles[3] + 360

        # Calc the average rotation
        rotation_angle = np.mean(rotation_angles)

        if rotation_angle == -180:
            rotation_angle = 180

    return rotation_angle


def dilate_and_crop_image_and_mask(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float = 0.0
) -> tuple[ScanImage, MaskArray]:
    if rotation_angle != 0.0:
        dilate_steps = 3
        mask = (mask > 0.5).astype(float)
        mask = binary_dilation(mask, iterations=dilate_steps).astype(float)

    scan_image_cropped = ScanImage(
        data=crop_to_mask(scan_image.data, mask),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
    mask_cropped = crop_to_mask(mask, mask)

    return scan_image_cropped, mask_cropped


def rotate_scan_image(scan_image: ScanImage, rotation_angle: float = 0.0) -> ScanImage:
    """Rotate scan image by given angle."""
    data = scan_image.data.copy()
    data[np.isnan(data)] = 0.0
    # TODO: do we want to reshape to fit the entire image rotated, or crop to original size
    data_rotated = rotate(data, rotation_angle, reshape=True)
    # TODO: scipy rotate interpolates, which changes values
    data_rotated[np.where(data_rotated == 0.0)] = np.nan
    return ScanImage(
        data=data_rotated,
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
        meta_data=scan_image.meta_data,
    )


def rotate_crop_image(
    scan_image: ScanImage,
    mask: MaskArray,
    rotation_angle: float = 0.0,
    crop_info: list[CropInfo] | None = None,
    times_median: int = 15,
) -> tuple[ScanImage, MaskArray]:
    # TODO: maskarray is zeros and ones, not boolean. gives errors
    rotation_angle = get_rotation_angle(scan_image, rotation_angle, crop_info)
    scan_image_dilated, mask_dilated = dilate_and_crop_image_and_mask(
        scan_image, mask, rotation_angle
    )
    scan_image_cleaned = remove_needles(scan_image_dilated, mask_dilated, times_median)
    scan_image_masked = mask_2d_array(scan_image_cleaned.data, mask_dilated)
    scan_image_rotated = rotate_scan_image(scan_image_masked, rotation_angle)
    return scan_image_rotated, mask_dilated
