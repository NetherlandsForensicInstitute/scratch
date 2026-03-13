import numpy as np
from loguru import logger
from scipy.ndimage import binary_dilation, rotate

from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.mask import crop_to_mask
from conversion.utils import update_scan_image_data


def get_rotation_angle(bounding_box: BoundingBox) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle by computing the angles between edges and the x-axis, and selecting the angle with
    the smallest absolute value.

    :param bounding_box: Bounding box of a rectangular crop region. Expects pixel coordinates,
        i.e. top-left origin, in the order [x, y].
    :return: The rotation angle in degrees, ranging from -180 to 180 (inclusive).
    """
    angles = []
    for i in range(4):
        point1 = bounding_box[i]
        point2 = bounding_box[(i + 1) % 4]
        angles.append(
            np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        )

    # find smallest absolute angle
    rotation_angle = min(angles, key=lambda x: abs(x))

    return rotation_angle


def crop_image_and_mask_to_mask(
    scan_image: ScanImage, mask: BinaryMask, margin: int
) -> tuple[ScanImage, BinaryMask]:
    """
    Crop scan_image.data and the mask itself to the bounding box of the mask. If a margin is given, the bounding box
    will be expanded (in case of a negative margin) or cropped (in case of a positive margin) by that amount.

    :param scan_image: Scan image to crop.
    :param mask: Binary mask array.
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box.
    :return: Tuple of the cropped scan_image and mask.
    """
    scan_image_cropped = update_scan_image_data(
        scan_image, crop_to_mask(scan_image.data, mask, margin)
    )
    mask_cropped = crop_to_mask(mask.astype(float), mask, margin).astype(bool)

    return scan_image_cropped, mask_cropped


def rotate_mask(mask: BinaryMask, rotation_angle: float) -> BinaryMask:
    """
    Rotate mask if rotation angle is not 0.

    :param mask: Binary mask indicating the crop region.
    :param rotation_angle: Rotation angle in degrees. Positive values rotate counterclockwise. If 0, returns inputs
        unchanged.
    :return: BinaryMask of  mask.
    """
    if np.isclose(rotation_angle, 0.0):
        return mask

    mask = rotate(
        mask,
        rotation_angle,
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)

    return mask
