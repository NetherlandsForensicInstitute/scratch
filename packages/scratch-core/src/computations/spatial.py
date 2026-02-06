import numpy as np
from container_models.base import BinaryMask, FloatArray2D
from numpy.typing import NDArray
from skimage.transform import resize


def get_bounding_box(mask: BinaryMask) -> tuple[slice, slice]:
    """
    Compute the minimal bounding box of a 2D mask.

    Finds the smallest axis-aligned rectangle containing all non-zero (or True) values.

    :param mask: 2D mask (non-zero/True values indicate the region of interest)
    :returns: Tuple (y_slice, x_slice) as slices for NumPy indexing, covering all mask pixels
    """
    coordinates = np.nonzero(mask)
    y_min, x_min = np.min(coordinates, axis=1)
    y_max, x_max = np.max(coordinates, axis=1)
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1)

def resample_array[T:BinaryMask|FloatArray2D](array:T,height:float,width:float, anti_aliasing:bool)->T:
    """
    Resample a 2D array to a new resolution.

    This function wraps `skimage.transform.resize`.

    :param array: Input 2D array (binary mask or float array)
    :param height: Target height in pixels
    :param width: Target width in pixels
    :param anti_aliasing: Whether to apply anti-aliasing during resizing
    :returns: Resampled array
    """
    return resize(
        image=array,
        output_shape=(height,width),
        mode="edge",
        anti_aliasing=anti_aliasing
    )