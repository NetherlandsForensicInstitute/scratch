import numpy as np
from numpy._typing import NDArray
from scipy.signal import convolve2d

from surface_conversion.schemas import LightAngle


def compute_surface_normals(
    depthdata: NDArray[tuple[int, int]],
    xdim: int,
    ydim: int,
    kernel: NDArray = np.array([[0, 1j, 0], [1, 0, -1], [0, -1j, 0]]),
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
    kernel : NDArray
        the kernel used to convolve the diff of the neighboring cells

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
    z = convolve2d(depthdata, kernel, "same", fillvalue=np.nan)
    hx = z.real * factor_x
    hy = z.imag * factor_y

    norm = np.sqrt(hx * hx + hy * hy + 1)
    n1 = -hx / norm
    n2 = hy / norm
    n3 = 1 / norm
    return n1, n2, n3


def calculate_lighting(
    light_vector: NDArray,
    observer_vector: NDArray,
    n1: NDArray,
    n2: NDArray,
    n3: NDArray,
    specular_factor: float = 1.0,
    phong_exponent: int = 4,
) -> NDArray:
    """
    Calculate diffuse + specular lighting components.

    Parameters
    ----------
    light_vector : NDArray
        Direction to light source (normalized).
    observer_vector : NDArray
        Direction to observer (normalized).
    n1, n2, n3 : NDArray
        Surface normal components.
    specular_factor : float
        Weight of specular component (default: 1.0).
    phong_exponent : int
        Phong exponent for specular highlights (default: 4).

    Returns
    -------
    NDArray
        Combined lighting intensity [0, 1].
    """
    h = light_vector + observer_vector
    h = h / np.linalg.norm(h)

    diffuse = light_vector[0] * n1 + light_vector[1] * n2 + light_vector[2] * n3
    diffuse = np.maximum(diffuse, 0)

    specular = h[0] * n1 + h[1] * n2 + h[2] * n3
    specular = np.maximum(specular, 0)
    specular = np.cos(2 * np.arccos(specular))
    specular = np.maximum(specular, 0) ** phong_exponent

    intensity = (diffuse + specular_factor * specular) / (1 + specular_factor)
    return intensity


def apply_multiple_lights(
    n1: NDArray[tuple[int, int]],
    n2: NDArray[tuple[int, int]],
    n3: NDArray[tuple[int, int]],
    light_angles: tuple[LightAngle],
    observer: LightAngle = LightAngle(azimuth=0, elevation=90),
):
    """

    Parameters
    ----------
    depthdata
    n1
    n2
    n3
    light_angles

    Returns
    -------

    """
    return np.stack(
        [
            calculate_lighting(light_angle.vector, observer.vector, n1, n2, n3)
            for light_angle in light_angles
        ],
        axis=-1,
    )
