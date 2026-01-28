"""
Utility functions for filtering operations.
"""

import numpy as np
from numpy.typing import NDArray

from conversion.mask import _determine_bounding_box


def remove_zero_border(
    data: NDArray[np.floating], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Remove zero/invalid borders from masked data.

    Finds the bounding box of valid (non-NaN, masked) data and crops to that region.

    :param data: 2D data array (may contain NaN).
    :param mask: Boolean mask (True = valid data).
    :returns: Tuple of (cropped_data, cropped_mask).
    """
    # Consider both mask and NaN values when finding valid region
    valid_data = mask & ~np.isnan(data)

    if not np.any(valid_data):
        # No valid data at all - return empty arrays
        return (
            np.array([]).reshape(0, data.shape[1]),
            np.array([], dtype=bool).reshape(0, data.shape[1]),
        )

    y_slice, x_slice = _determine_bounding_box(valid_data)

    return data[y_slice, x_slice], mask[y_slice, x_slice]
