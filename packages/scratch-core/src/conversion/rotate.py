import numpy as np
from scipy.ndimage import binary_dilation, rotate

from conversion.utils import update_scan_image_data
from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import mask_and_remove_needles
from conversion.data_formats import RectangularBoundingBox
from conversion.mask import crop_to_mask

# Number of iterations to dilate a mask before it is rotated
DILATE_STEPS = 3


def rotate_crop_and_mask_image_by_crop(
    scan_image: ScanImage,
    mask: MaskArray,
    rectangular_bounding_box: RectangularBoundingBox | None,
    median_factor: float = 15,
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
    :param rectangular_bounding_box: Bounding box of a rectangular crop region used to determine the rotation of an
        image, or None. Expects pixel coordinates, i.e. top-left origin.
    :param median_factor: Parameter used to determine what is considered an outlier when removing outliers/needles.
    :return: The cropped, rotated and masked scan image.
    """
    rotation_angle = (
        get_rotation_angle(rectangular_bounding_box)
        if rectangular_bounding_box is not None
        else 0.0
    )

    margin = 0
    if not np.isclose(rotation_angle, 0.0):
        mask = binary_dilation(mask, iterations=DILATE_STEPS).astype(bool)
        # Define a margin to reverse dilation later on
        margin = DILATE_STEPS + 2

    scan_image_cropped, mask_cropped = crop_image_and_mask_to_mask(
        scan_image, mask, margin
    )

    scan_image_cleaned_and_masked = mask_and_remove_needles(
        scan_image_cropped, mask_cropped, median_factor
    )

    # Rotate by the negative rotation_angle to counter the current rotation
    scan_image_rotated, mask_rotated = rotate_mask_and_scan_image(
        scan_image_cleaned_and_masked, mask_cropped, -rotation_angle
    )

    scan_image_cropped = update_scan_image_data(
        scan_image, crop_to_mask(scan_image_rotated.data, mask_rotated, margin)
    )
    return scan_image_cropped


def get_rotation_angle(rectangular_bounding_box: RectangularBoundingBox) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle by computing the angles between following points (e.g. top left corner with top
    right, top right with bottom right) and selecting the angle with the smallest absolute value.

    :param rectangular_bounding_box: Bounding box of a rectangular crop region. Expects pixel coordinates,
        i.e. top-left origin.
    :return: The rotation angle in degrees, ranging from -180 to 180 (inclusive).
    """
    angles = []
    for i in range(4):
        point1 = rectangular_bounding_box[i]
        point2 = rectangular_bounding_box[(i + 1) % 4]
        angles.append(
            np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        )

    # find smallest absolute angle
    rotation_angle = min(angles, key=lambda x: abs(x))

    return rotation_angle


def crop_image_and_mask_to_mask(
    scan_image: ScanImage, mask: MaskArray, margin: int
) -> tuple[ScanImage, MaskArray]:
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


def rotate_mask_and_scan_image(
    scan_image: ScanImage, mask: MaskArray, rotation_angle: float
) -> tuple[ScanImage, MaskArray]:
    """
    Rotate mask and scan image if rotation angle is not 0.

    :param scan_image: Image to rotate and crop.
    :param mask: Binary mask indicating the crop region.
    :param rotation_angle: Rotation angle in degrees. Positive values rotate counterclockwise. If 0, returns inputs
        unchanged.
    :return: Tuple of (rotated) scan image and mask.
    """
    if np.isclose(rotation_angle, 0.0):
        return scan_image, mask

    mask = rotate(
        mask,
        rotation_angle,
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
    ).astype(bool)

    scan_image = update_scan_image_data(
        scan_image,
        rotate(
            scan_image.data,
            rotation_angle,
            reshape=True,
            order=1,
            mode="constant",
            cval=np.nan,
        ),
    )
    return scan_image, mask
