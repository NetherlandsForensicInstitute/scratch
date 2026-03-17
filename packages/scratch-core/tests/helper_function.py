from functools import partial

import numpy as np

from container_models.scan_image import ScanImage


def assert_nan_mask_preserved(
    input_array: np.ndarray, output_array: np.ndarray
) -> None:
    """Assert that NaN locations in input are still NaN in output, and vice versa."""
    if input_array.ndim > output_array.ndim:
        input_nan = np.isnan(input_array[..., 0])
    else:
        input_nan = np.isnan(input_array)

    if output_array.ndim > input_array.ndim:
        output_nan = np.isnan(output_array[..., 0])
    else:
        output_nan = np.isnan(output_array)

    np.testing.assert_array_equal(input_nan, output_nan)


NoScaleScanImage = partial(ScanImage, scale_x=1, scale_y=1)
