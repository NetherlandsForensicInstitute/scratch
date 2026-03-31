import numpy as np

from container_models.base import BinaryMask


def get_bounding_box(mask: BinaryMask, margin: int) -> tuple[slice, slice]:
    """
    Compute the minimal bounding box of a 2D mask.

    Finds the smallest axis-aligned rectangle containing all non-zero (or True) values.

    :param mask: 2D mask (non-zero/True values indicate the region of interest)
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :returns: Tuple (y_slice, x_slice) as slices for bounding_box.
    """
    y_coords, x_coords = np.nonzero(mask)
    y_min = max(0, y_coords.min() + margin)
    y_max = min(mask.shape[0], y_coords.max() - margin + 1)
    x_min = max(0, x_coords.min() + margin)
    x_max = min(mask.shape[1], x_coords.max() - margin + 1)

    if x_min >= x_max:
        raise ValueError("Slice results in x_min >= x_max. Margin may be too large.")
    if y_min >= y_max:
        raise ValueError("Slice results in y_min >= y_max. Margin may be too large.")

    return slice(y_min, y_max), slice(x_min, x_max)
