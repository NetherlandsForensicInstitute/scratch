from container_models.base import BinaryMask
import numpy as np


def get_bounding_box(mask: BinaryMask) -> tuple[slice, slice]:
    """
    Compute the minimal bounding box of a 2D mask.

    Finds the smallest axis-aligned rectangle containing all non-zero (or True) values.

    :param mask: 2D mask (non-zero/True values indicate the region of interest)
    :returns: Tuple (y_slice, x_slice) as slices for NumPy indexing, covering all mask pixels
    """
    coordinates = np.nonzero(mask)
    y_min, x_min = np.min(coordinates, axis=1)
    y_max, x_max = np.max(coordinates, axis=1)
    return slice(x_min, x_max + 1), slice(y_min, y_max + 1)
