import numpy as np
from numpy.typing import NDArray
from image_generation.data_formats import ScanImage


def prepare_2d_grid(
    scan_image: ScanImage, image_center: tuple[float, float] | None = None
) -> tuple[NDArray, NDArray]:
    """
    Return a 2D grid containing the physical coordinates of the scan data relative to the image center.

    :param scan_image: An instance of `ScanImage` containing the recorded depth data.
    :param image_center: The physical coordinates of the center of the image. If `None`,
        then the image center will be computed from the scan data.
    :returns: A tuple containing the grid coordinates for the X-direction and Y-direction.
    """
    # Generate Grid (ij indexing to match matrix coordinates)
    x_indices, y_indices = np.meshgrid(
        np.arange(scan_image.width), np.arange(scan_image.height), indexing="ij"
    )
    # Center grid
    if not image_center:
        image_center = (
            (scan_image.width - 1)
            * scan_image.scale_x
            * 0.5,  # X-coordinate of image center
            (scan_image.height - 1)
            * scan_image.scale_y
            * 0.5,  # Y-coordinate of image center
        )
    # Translate the grid so that the image center lies in the origin
    x_grid = (x_indices * scan_image.scale_x) - image_center[0]
    y_grid = (y_indices * scan_image.scale_y) - image_center[1]

    return x_grid, y_grid
