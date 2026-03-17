import numpy as np
from scipy.ndimage import rotate

from container_models.base import BinaryMask


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
