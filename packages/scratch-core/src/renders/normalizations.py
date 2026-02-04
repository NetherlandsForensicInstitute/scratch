import numpy as np
from typing import Final
from container_models.base import DepthData, FloatArray2D, Scale, VectorField


# Padding configurations for gradient arrays to maintain original dimensions
_PAD_X_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (1, 1),
)  # Pad left and right (columns)
_PAD_Y_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (0, 0),
)  # Pad top and bottom (rows)
MAX_SCALE: Final[int] = 255
MIN_SCALE: Final[int] = 25


def _pad_gradient(
    unpadded_gradient: FloatArray2D, pad_width: tuple[tuple[int, int], tuple[int, int]]
) -> FloatArray2D:
    """Pad a gradient array with NaN values at the borders."""
    return np.pad(unpadded_gradient, pad_width, mode="constant", constant_values=np.nan)


def normalize_to_surface_normals(data: DepthData, scale: Scale) -> VectorField:
    """Normalize gradient components to unit surface normal vectors."""
    gradient_x = _pad_gradient(
        (data[:, :-2] - data[:, 2:]) * scale.x,
        _PAD_X_GRADIENT,
    )
    gradient_y = _pad_gradient(
        (data[:-2, :] - data[2:, :]) * scale.y,
        _PAD_Y_GRADIENT,
    )
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + 1)
    return np.stack(
        [gradient_x / magnitude, -gradient_y / magnitude, 1 / magnitude], axis=-1
    )


def normalize_2d_array(data: VectorField) -> DepthData:
    """
    Normalize a 2D intensity map to a specified output range.

    The normalization is done by the steps:
    1. apply min-max normalization to grayscale data
    2. stretch / scale the normalized data from the unit range to a specified output range

    :param image_to_normalize: 2D array of input intensity values.

    :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
    """
    imin = np.nanmin(data)
    imax = np.nanmax(data)
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (data - imin) / (imax - imin)
