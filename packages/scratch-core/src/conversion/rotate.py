import numpy as np
from scipy.ndimage import rotate

from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.data_formats import BoundingBox
from conversion.mask import crop_to_mask
from conversion.utils import update_scan_image_data


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
