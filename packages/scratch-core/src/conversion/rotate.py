import numpy as np
from scipy.ndimage import binary_dilation, rotate

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import remove_needles
from conversion.data_formats import CropType, CropInfo
from conversion.mask import crop_to_mask, mask_2d_array


def get_rotation_angle(
    rotation_angle: float = 0.0,
    crop_info: list[CropInfo] = None,
) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle either from an explicitly provided angle or by
    computing it from the corner points of a rectangular crop. When computing from
    corners, the function orders the points counter-clockwise, analyzes the angles
    of each side relative to the expected orientation of a non-rotated rectangle,
    and returns the average rotation angle.

    :param rotation_angle: Explicit rotation angle in degrees. If non-zero, this
                          value is returned directly without computation.
    :param crop_info: List of crop information objects. If provided and the first
                     crop is of type RECTANGLE, the rotation angle is computed
                     from the corner points in the crop data.
    :return: The rotation angle in degrees, ranging from -180 to 180 (exclusive
            of -180, which is normalized to 180).
    """
    if (
        rotation_angle == 0.0
        and crop_info
        and crop_info[0].crop_type == CropType.RECTANGLE
    ):
        points = crop_info[0].data["corner"]

        # Sort points counter-clockwise by angle from center
        mean_xy = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - mean_xy[1], points[:, 0] - mean_xy[0])
        points_ordered = points[np.argsort(angles)]

        index = np.where(np.all(points_ordered == points[0, :], axis=1))[0][0]
        points_ordered = np.roll(points_ordered, shift=-index, axis=0)

        # Determine the rotation angle
        P1, P2, P3, P4 = points_ordered
        dx = np.array([P2[1] - P1[1], P3[1] - P2[1], P3[1] - P4[1], P4[1] - P1[1]])
        dy = np.array([P2[0] - P1[0], P3[0] - P2[0], P3[0] - P4[0], P4[0] - P1[0]])
        rotation_angles = np.degrees(np.arctan2(dx, dy))

        # A rectangle that was not rotated should have resulted in [0, -90, 0, -90]
        # The angle over which each side of the rectangle is rotated is
        rotation_angles = rotation_angles - np.array([0, -90, 0, -90])

        # The 4 rotation angles should be almost similar, but due
        # to the fact that arctan2 jumps from -180 to 180 degrees, some strange results may occur.
        # Therefore, the following checks are performed
        rotation_angles = np.unwrap(rotation_angles, period=360)
        rotation_angle = np.mean(rotation_angles)
        if rotation_angle == -180:
            rotation_angle = 180

    return rotation_angle


def dilate_and_crop_image_and_mask(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float = 0.0
) -> tuple[ScanImage, MaskArray]:
    if rotation_angle != 0.0:
        dilate_steps = 3
        mask = binary_dilation(mask, iterations=dilate_steps)

    scan_image_cropped = ScanImage(
        data=crop_to_mask(scan_image.data, mask),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
    mask_cropped = crop_to_mask(mask, mask).astype(bool)

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
    times_median: float = 15,
) -> tuple[ScanImage, MaskArray]:
    rotation_angle = get_rotation_angle(scan_image, rotation_angle, crop_info)
    scan_image_dilated, mask_dilated = dilate_and_crop_image_and_mask(
        scan_image, mask, rotation_angle
    )
    scan_image_cleaned = remove_needles(scan_image_dilated, mask_dilated, times_median)
    scan_image_masked = ScanImage(
        data=mask_2d_array(scan_image_cleaned.data, mask_dilated),
        scale_x=scan_image_cleaned.scale_x,
        scale_y=scan_image_cleaned.scale_y,
    )
    scan_image_rotated = rotate_scan_image(scan_image_masked, rotation_angle)
    return scan_image_rotated, mask_dilated


def plot_rectangle(corners: list[tuple[float, float]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Close the rectangle by adding the first point at the end
    corners_closed = np.vstack([corners, corners[0]])

    plt.plot(corners_closed[:, 0], corners_closed[:, 1], "r-", linewidth=2)
    plt.plot(corners[:, 0], corners[:, 1], "bo", markersize=8)  # Show corner points
    plt.axis("equal")
    plt.grid(True)
    plt.show()
