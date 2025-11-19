import numpy as np
from numpy._typing import NDArray
from scipy.signal import convolve2d


def convert_image_to_slope_map(
    depthdata: NDArray[tuple[int, int]],
    xdim: int,
    ydim: int,
    kernel: tuple[int, int] = (
        [0, 1j, 0],
        [1, 0, -1],
        [0, -1j, 0],
    ),
):
    """
    Compute surface-normal components (n1, n2, n3) from a 2D depth map.

    Parameters
    ----------
    depthdata : NDArray
        2-D array of depth values, indexed as depth[y, x].
    xdim : float
        Physical spacing between columns in meters (Δx).
    ydim : float
        Physical spacing between rows in meters (Δy).

    Returns
    -------
    n1, n2, n3 : NDArray
        Components of the unit surface normal.
        n1 = -∂z/∂x / norm
        n2 = -∂z/∂y / norm
        n3 = 1 / norm
    """
    factor_x = 1 / (2 * xdim)
    factor_y = 1 / (2 * ydim)
    z = convolve2d(depthdata, np.array(kernel), "same", fillvalue=np.nan)
    hx = z.real * factor_x
    hy = z.imag * factor_y

    norm = np.sqrt(hx * hx + hy * hy + 1)
    n1 = -hx / norm
    n2 = hy / norm
    n3 = 1 / norm
    return n1, n2, n3
