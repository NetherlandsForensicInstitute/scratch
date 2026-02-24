"""
Shared utility functions for the CMC surface comparison pipeline.
"""

import numpy as np

from container_models.base import FloatArray1D


def um_to_pixels(size_um: FloatArray1D, pixel_spacing: FloatArray1D) -> np.ndarray:
    """
    Convert a size in micrometers to the nearest integer number of pixels.

    :param size_um: Size [width, height] in micrometers, shape (2,).
    :param pixel_spacing: Pixel spacing [dx, dy] in micrometers, shape (2,).
    :returns: Size [pixel_width, pixel_height] in pixels as integers, shape (2,).
    """
    return np.round(size_um / pixel_spacing).astype(int)


def center_um_to_top_left_pixel(
    center_um: FloatArray1D,
    cell_size_px: np.ndarray,
    pixel_spacing: FloatArray1D,
) -> tuple[int, int]:
    """
    Convert a cell center in micrometers to the top-left corner pixel index.

    :param center_um: Cell center [x, y] in micrometers, shape (2,).
    :param cell_size_px: Cell size [pixel_width, pixel_height] in pixels, shape (2,).
    :param pixel_spacing: Pixel spacing [dx, dy] in micrometers, shape (2,).
    :returns: ``(row, col)`` of the top-left corner of the cell (may be negative).
    """
    row = int(round(center_um[1] / pixel_spacing[1] - cell_size_px[1] / 2))
    col = int(round(center_um[0] / pixel_spacing[0] - cell_size_px[0] / 2))
    return row, col
