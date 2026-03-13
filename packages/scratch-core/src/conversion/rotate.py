import numpy as np
from loguru import logger
from scipy.ndimage import binary_dilation, rotate

from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.mask import crop_to_mask
from conversion.utils import update_scan_image_data
from mutations import Mask, Rotate
from mutations.filter import FilterNeedles
from mutations.spatial import CropToMask


def rotate_and_crop_by_mask_crop(
    scan_image: ScanImage,
    mask: BinaryMask,
    bounding_box: BoundingBox,
) -> ScanImage:
    """
    Rotates, crops and masks a scan image based on the given mask and rectangle.

    Implements the following flow:
    - Determine the rotation angle for the image and mask by the bounding box of rectangle, if a rectangle is given.
        Otherwise, the rotation angle is 0.
    - If the rotation angle is not 0, the mask is binary dilated using DILATE_STEPS iterations to correct for
        imperfections when rotating. A margin is determined to reduce the final image to compensate for the dilation.
    - The mask and image are cropped to the bounds of the mask.
    - The scan image is masked using the cropped mask and cleaned of needles (i.e. steep slopes). The parameter
        `times_median` is used to determine the threshold to find outliers.
    - The image and mask are rotated by rotation_angle.
    - The rotated image is cropped to the bounds of the rotated mask, using margin to compensate for dilation.

    :param scan_image: Scan image to rotate, mask and crop.
    :param mask: Binary mask array.
    :param bounding_box: Bounding box of a rectangular crop region used to determine the rotation of an
        image, or None. Expects pixel coordinates, i.e. top-left origin.
    :return: The cropped, rotated and masked scan image.
    """
    margin = 0
    rotator = Rotate.from_bounding_box(bounding_box=bounding_box)
    if not np.isclose(rotator.rotation_angle, 0.0):
        dilate_steps = 3
        margin = dilate_steps + 2
        logger.debug("Rotating mask")
        mask = rotate_mask(
            mask=binary_dilation(mask, iterations=dilate_steps).astype(bool),
            rotation_angle=rotator.rotation_angle,
        )
        logger.debug("Rotating image")
        scan_image = rotator(scan_image=scan_image).unwrap()
    logger.debug(f"Cropping image with margin: {margin}")
    return CropToMask(mask=mask, margin=margin)(scan_image).unwrap()


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
