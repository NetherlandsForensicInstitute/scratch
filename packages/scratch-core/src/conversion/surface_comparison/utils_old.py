"""
Shared utility functions for the CMC surface comparison pipeline.
"""

import numpy as np

from container_models.base import FloatArray1D


def meters_to_pixels(size_m: FloatArray1D, pixel_spacing: FloatArray1D) -> np.ndarray:
    """
    Convert a size in meters to the nearest integer number of pixels.

    :param size_m: Size [width, height] in meters, shape (2,).
    :param pixel_spacing: Pixel spacing [dx, dy] in meters, shape (2,).
    :returns: Size [pixel_width, pixel_height] in pixels as integers, shape (2,).
    """
    return np.round(size_m / pixel_spacing).astype(int)
