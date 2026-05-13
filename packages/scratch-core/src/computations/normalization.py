import numpy as np

from container_models.base import FloatArray2D, ImageRGBA
from container_models.models import NormalizationBounds


def _normalize_2d_array(
    array_to_normalize: FloatArray2D, normalization_bounds: NormalizationBounds
) -> FloatArray2D:
    """
    Normalize a 2D intensity map to a specified output range.
    The normalization is done by the steps:
    1. apply min-max normalization to grayscale data
    2. stretch / scale the normalized data from the unit range to a specified output range

    :note: If all valid pixels have the same value (no contrast), the output
    is filled with the midpoint of the output range. NaN pixels are preserved.

    :param array_to_normalize: 2D array of input intensity values.
    :param normalization_bounds: the scaling for normalization.
    :returns: Normalized 2D intensity map with values in `[normalization_bounds.low, normalization_bounds.high]``.
    """
    imin = np.nanmin(array_to_normalize)
    imax = np.nanmax(array_to_normalize)

    if imax == imin:
        fill_value = (normalization_bounds.low + normalization_bounds.high) / 2
        result = np.full_like(array_to_normalize, fill_value)
        result[np.isnan(array_to_normalize)] = np.nan
        return result

    norm = (array_to_normalize - imin) / (imax - imin)
    return (
        normalization_bounds.low
        + (normalization_bounds.high - normalization_bounds.low) * norm
    )


def _grayscale_to_rgba(scan_data: FloatArray2D) -> ImageRGBA:
    """
    Convert a 2D grayscale array to an 8-bit RGBA array.

    The grayscale pixel values are assumed to be floating point values in the [0, 255] interval.
    NaN values will be converted to black pixels with 100% transparency.

    :param scan_data: The grayscale image data to be converted to an 8-bit RGBA image.
    :returns: Array with the image data in 8-bit RGBA format.
    """
    gray_uint8 = np.nan_to_num(scan_data, nan=0.0).astype(np.uint8)
    rgba = np.repeat(gray_uint8[..., np.newaxis], 4, axis=-1)
    rgba[..., 3] = (~np.isnan(scan_data)).astype(np.uint8) * 255
    return rgba
