from container_models.base import ScanMap2DArray, MaskArray
from conversion.mask import _determine_bounding_box


def crop_nan_borders(data: ScanMap2DArray, mask: MaskArray) -> ScanMap2DArray:
    """
    Crop 2D array by removing NaN border rows and columns.

    :param data: Input 2D array.
    :param mask: Valid data mask.
    :return: cropped array
    """
    x_slice, y_slice = _determine_bounding_box(mask)
    return data[y_slice, x_slice]
