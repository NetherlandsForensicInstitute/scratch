import numpy as np

from container_models.base import FloatArray2D, Pair
from container_models import ImageContainer


def get_2d_grid(
    scan_image: ImageContainer, offset: Pair[float] = Pair(0, 0)
) -> Pair[FloatArray2D]:
    """
    Return a 2D grid containing the physical coordinates of the scan data.

    :param scan_image: An instance of `ScanImage` containing the recorded depth data.
    :param offset: A tuple containing the physical coordinates of the image offset (in meters)
        relative to the origin by which the grid coordinates need to be translated. The first element
        corresponds to offset in the X-dimension, and the second element to the offset in the Y-dimension.
    :returns: A tuple containing the grid coordinates for the X-direction and Y-direction.
    """
    return Pair(  # type: ignore[return-value]
        *[
            # Translate the grid by `offset`
            (indices * scale) + offset_
            for indices, scale, offset_ in zip(
                # Generate Grid (ij indexing to match matrix coordinates)
                np.meshgrid(
                    np.arange(scan_image.width),
                    np.arange(scan_image.height),
                    indexing="xy",
                ),
                scan_image.metadata.scale,
                offset,
            )
        ]
    )
