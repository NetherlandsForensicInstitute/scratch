import numpy as np
from numpy.typing import NDArray
from image_generation.data_formats import ScanImage


def get_2d_grid(
    scan_image: ScanImage, offset: tuple[float, float] | None = None
) -> tuple[NDArray, NDArray]:
    """
    Return a 2D grid containing the physical coordinates of the scan data relative to the image center.

    :param scan_image: An instance of `ScanImage` containing the recorded depth data.
    :param offset: A tuple containing the physical coordinates of the image offset (in meters) relative to the
        origin. The first element corresponds to offset in the X-dimension, and the second element to the
        offset in the Y-dimension. If `None`, then the center of the scan image will be taken as offset
        so that the coordinates will be centered around the origin.
    :returns: A tuple containing the grid coordinates for the X-direction and Y-direction.
    """
    # Generate Grid (ij indexing to match matrix coordinates)
    x_indices, y_indices = np.meshgrid(
        np.arange(scan_image.width), np.arange(scan_image.height), indexing="ij"
    )

    if not offset:
        offset = (
            (scan_image.width - 1)
            * scan_image.scale_x
            * 0.5,  # X-coordinate of image center
            (scan_image.height - 1)
            * scan_image.scale_y
            * 0.5,  # Y-coordinate of image center
        )
    # Translate the grid by `offset`
    x_grid = (x_indices * scan_image.scale_x) - offset[0]
    y_grid = (y_indices * scan_image.scale_y) - offset[1]

    return x_grid, y_grid
