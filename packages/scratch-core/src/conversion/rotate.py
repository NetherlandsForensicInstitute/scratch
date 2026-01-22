import numpy as np
from scipy.ndimage import binary_dilation, rotate

from conversion.utils import update_scan_image_data
from container_models.base import MaskArray
from container_models.scan_image import ScanImage
from conversion.remove_needles import mask_and_remove_needles
from conversion.data_formats import RectangularCrop
from conversion.mask import crop_to_mask

# Number of iterations to dilate a mask before it is rotated
DILATE_STEPS = 3


def rotate_crop_and_mask_image_by_crop(
    scan_image: ScanImage,
    mask: MaskArray,
    rectangle: RectangularCrop | None,
    times_median: float = 15,
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
    :param rectangle: Bounding box of a rectangular crop region used to determine the rotation of an image, or None.
    :param times_median: Parameter used to determine what is considered an outlier when removing outliers/needles.
    :return: The cropped, rotated and masked scan image.
    """
    rotation_angle = get_rotation_angle(rectangle) if rectangle is not None else 0.0

    margin = 0
    if rotation_angle != 0.0:
        mask = binary_dilation(mask, iterations=DILATE_STEPS).astype(bool)
        # Define a margin to reverse dilation later on
        margin = DILATE_STEPS + 2

    scan_image_cropped, mask_cropped = crop_image_and_mask_to_mask(
        scan_image, mask, margin
    )

    scan_image_cleaned_and_masked = mask_and_remove_needles(
        scan_image_cropped, mask_cropped, times_median
    )

    scan_image_rotated, mask_rotated = rotate_mask_and_scan_image(
        scan_image_cleaned_and_masked, mask_cropped, rotation_angle
    )

    scan_image_cropped = update_scan_image_data(
        scan_image, crop_to_mask(scan_image_rotated.data, mask_rotated, margin)
    )
    return scan_image_cropped


def get_rotation_angle(rectangle: RectangularCrop) -> float:
    """
    Calculate the rotation angle of a rectangular crop region.

    Determines the rotation angle by computing it from the corner points of a rectangular crop. When computing from
    corners, the function calculates the angle of each edge of the rectangle relative to the horizontal axis, then
    selects the edge that is closest to horizontal (smallest absolute angle). This angle is then normalized to the
    range [-90, 90] degrees.

    :param rectangle: Bounding box of a rectangular crop region.
    :return: The rotation angle in degrees, ranging from -90 to 90 (inclusive). The angle is normalized to this range
             to represent the minimal rotation needed to align the rectangle. If the first crop is not a rectangle, an
             angle of 0.0 is returned.
    """
    angles = []
    for i in range(4):
        p1 = rectangle[i]
        p2 = rectangle[(i + 1) % 4]
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

        scan_image = update_scan_image_data(
            scan_image,
            rotate(
                scan_image.data,
                -rotation_angle,
                reshape=True,
                order=1,
                mode="constant",
                cval=np.nan,
            ),
        )
    return scan_image, mask
