import numpy as np
from numpy.typing import NDArray
from container_models.scan_image import ScanImage


def get_2d_grid(
    scan_image: ScanImage, offset: tuple[float, float] = (0, 0)
) -> tuple[NDArray, NDArray]:
    """
    Return a 2D grid containing the physical coordinates of the scan data.

    :param scan_image: An instance of `ScanImage` containing the recorded depth data.
    :param offset: A tuple containing the physical coordinates of the image offset (in meters)
        relative to the origin by which the grid coordinates need to be translated. The first element
        corresponds to offset in the X-dimension, and the second element to the offset in the Y-dimension.
    :returns: A tuple containing the grid coordinates for the X-direction and Y-direction.
    """
    # Generate Grid (ij indexing to match matrix coordinates)
    x_indices, y_indices = np.meshgrid(
        np.arange(scan_image.width), np.arange(scan_image.height), indexing="ij"
    )
    # Translate the grid by `offset`
    x_grid = (x_indices * scan_image.scale_x) + offset[0]
    y_grid = (y_indices * scan_image.scale_y) + offset[1]

    return x_grid, y_grid
