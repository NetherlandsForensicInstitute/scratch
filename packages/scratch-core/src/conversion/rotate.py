from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, rotate

from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import mask_and_remove_needles
from conversion.data_formats import CropType, CropInfo
from conversion.mask import crop_to_mask

# Number of iterations to dilate a mask before it is rotated
DILATE_STEPS = 3


def rotate_crop_and_mask_image_by_crop(
    scan_image: ScanImage,
    mask: MaskArray,
    crop_infos: tuple[CropInfo],
    rotation_angle: float = 0.0,
    times_median: float = 15,
) -> ScanImage:
    """
    Rotates, crops and masks a scan image based on the given mask and crop info.

    Implements the following flow:
    - Determine the rotation angle for the image and mask
    By using the given rotation_angle if rotation_angle != 0. Otherwise, if the first object of crop_info is a
    rectangle, then the rotation_angle of that rectangle is determined. Any other case will lead to a rotation angle
    of 0.
    - If the rotation angle is not 0, the mask is binary dilated using DILATE_STEPS iterations to correct for
    imperfections when rotating. A margin is determined to reduce the final image to compensate for the dilation.
    - The mask and image are cropped to the bounds of the mask.
    - The scan image is cleaned of needles. The parameter `times_median` is used to determine the threshold to find
    outliers.
    - The cleaned image is masked using the cropped mask.
    - The image and mask are rotated by rotation_angle.
    - The rotated image is cropped to the bounds of the rotated mask, using margin to compensate for dilation.

    :param scan_image: Scan image to rotate, mask and crop.
    :param mask: Binary mask array.
    :param rotation_angle: Angle with which to rotate the image.
    :param crop_infos: List of crop info objects that describe the crops the user has done and the order of the crops.
    :param times_median: Parameter used to determine what is considered an outlier when removing outliers/needles.
    :return: The cropped, rotated and masked scan image.
    """
    rotation_angle = get_rotation_angle(crop_infos, rotation_angle)

    margin = None
    if rotation_angle != 0.0:
        mask = binary_dilation(mask, iterations=DILATE_STEPS).astype(bool)
        # Define a margin to reverse dilation later on
        margin = DILATE_STEPS + 2

    scan_image_cropped, mask_cropped = crop_image_and_mask_to_mask(scan_image, mask)

    scan_image_cleaned_and_masked = mask_and_remove_needles(
        scan_image_cropped, mask_cropped, times_median
    )

    scan_image_rotated, mask_rotated = rotate_mask_and_scan_image(
        scan_image_cleaned_and_masked, mask_cropped, rotation_angle
    )

    scan_image_cropped = ScanImage(
        data=crop_to_mask(scan_image_rotated.data, mask_rotated, margin),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
    return scan_image_cropped


def get_rotation_angle(
    crop_infos: tuple[CropInfo], rotation_angle: float = 0.0
) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle either from an explicitly provided angle or by computing it from the corner points
    of a rectangular crop if the first object in crop_info is of type RECTANGLE. When computing from corners, the
    function calculates the angle of each edge of the rectangle relative to the horizontal axis, then selects the edge
    that is closest to horizontal (smallest absolute angle). This angle is then normalized to the range [-90, 90]
    degrees.

    :param rotation_angle: Explicit rotation angle in degrees. If non-zero, this value is returned directly without
                           computation.
    :param crop_infos: Tuple of crop information objects. If provided and the first crop is of type RECTANGLE, the
                      rotation angle is computed from the corner points in the crop data.
    :return: The rotation angle in degrees, ranging from -90 to 90 (inclusive). The angle is normalized to this range
             to represent the minimal rotation needed to align the rectangle.
    """
    if (
        rotation_angle == 0.0
        and crop_infos
        and crop_infos[0].crop_type == CropType.RECTANGLE
    ):
        corners = crop_infos[0].data["corner"]
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            angles.append((abs(angle), angle))

        # find smallest absolute angle
        rotation_angle = min(angles, key=lambda x: abs(x[0]))[1]

    # Normalize to [-90, 90] range
    if rotation_angle > 90:
        rotation_angle -= 180
    elif rotation_angle < -90:
        rotation_angle += 180

    return rotation_angle


def crop_image_and_mask_to_mask(
    scan_image: ScanImage, mask: MaskArray, margin: Optional[int] = None
) -> tuple[ScanImage, MaskArray]:
    """
    Crop scan_image.data and the mask itself to the bounding box of the mask. If a margin is given, the bounding box
    will be expanded (in case of a negative margin) or cropped (in case of a positive margin) by that amount.

    :param scan_image: Scan image to crop.
    :param mask: Binary mask array.
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box.
    :return: Tuple of the cropped scan_image and mask.
    """
    scan_image_cropped = ScanImage(
        data=crop_to_mask(scan_image.data, mask, margin),
        scale_x=scan_image.scale_x,
        scale_y=scan_image.scale_y,
    )
    mask_cropped = crop_to_mask(mask.astype(float), mask, margin).astype(bool)

    return scan_image_cropped, mask_cropped


def rotate_mask_and_scan_image(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float
) -> tuple[ScanImage, MaskArray]:
    """
    Rotate mask and scan image if rotation angle is not 0.

    :param scan_image: Image to rotate and crop.
    :param mask: Binary mask indicating the crop region.
    :param rotation_angle: Rotation angle in degrees. Positive values rotate clockwise. If 0, returns inputs unchanged.
    :return: Tuple of (rotated) scan image and mask.
    """
    if rotation_angle != 0:
        mask = rotate(
            mask.astype(float),
            -rotation_angle,
            reshape=True,
            order=0,
            mode="constant",
            cval=0,
        ).astype(bool)

        scan_image = ScanImage(
            data=rotate(
                scan_image.data,
                -rotation_angle,
                reshape=True,
                order=1,
                mode="constant",
                cval=np.nan,
            ),
            scale_x=scan_image.scale_x,
            scale_y=scan_image.scale_y,
        )
    return scan_image, mask
