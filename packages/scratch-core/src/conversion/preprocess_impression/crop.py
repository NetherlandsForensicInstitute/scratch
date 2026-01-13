import numpy as np

from container_models.base import ScanMap2DArray
from conversion.mask import _determine_bounding_box


def crop_nan_borders(data: ScanMap2DArray) -> ScanMap2DArray:
    """
    Crop 2D array by removing NaN border rows and columns.

    :param data: Input 2D array.
    :return: cropped array
    """
    mask = ~np.isnan(data)
    x_slice, y_slice = _determine_bounding_box(mask)
    return data[y_slice, x_slice]
