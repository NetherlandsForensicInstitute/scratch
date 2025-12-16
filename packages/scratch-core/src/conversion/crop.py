import numpy as np

from utils.array_definitions import ScanMap2DArray


def crop_nan_borders(data: ScanMap2DArray) -> ScanMap2DArray:
    """
    Crop 2D array by removing NaN border rows and columns.

    :param data: Input 2D array.
    :return: cropped array
    """
    valid_mask = ~np.isnan(data)

    # Find rows and columns with any valid (non-NaN) data
    valid_rows = np.where(np.any(valid_mask, axis=1))[0]
    valid_cols = np.where(np.any(valid_mask, axis=0))[0]

    # Handle all-NaN case
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return np.array([[]])

    return data[valid_rows[0] : valid_rows[-1] + 1, valid_cols[0] : valid_cols[-1] + 1]
