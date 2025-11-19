import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, Field
from scipy.signal import convolve2d


class LightAngle(BaseModel):
    azimuth: float = Field(..., description="Azimuth angle in degrees.")
    elevation: float = Field(..., description="Elevation angle in degrees.")

    @property
    def vector(self):
        azr = np.deg2rad(self.azimuth)
        elr = np.deg2rad(self.elevation)
        v = np.array(
            [-np.cos(azr) * np.cos(elr), np.sin(azr) * np.cos(elr), np.sin(elr)]
        )
        return v


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


def calculate_surface(LS, OBS, n1, n2, n3):
    """Calculate diffuse + specular components."""
    # PREPARATIONS
    h = LS + OBS
    h = h / np.sqrt(np.dot(h, h))  # normalize

    # DIFFUSE COMPONENT
    Idiffuse = LS[0] * n1 + LS[1] * n2 + LS[2] * n3
    Idiffuse[Idiffuse < 0] = 0

    # SPECULAR COMPONENT
    Ispec = h[0] * n1 + h[1] * n2 + h[2] * n3
    Ispec[Ispec < 0] = 0
    Ispec = np.cos(2 * np.arccos(Ispec))
    Ispec[Ispec < 0] = 0

    # Phong factor f=4
    Ispec = Ispec**4

    # Combine
    fspec = 1
    Iout = (Idiffuse + fspec * Ispec) / (1 + fspec)
    return Iout


def merge_depth_map_with_slope_maps(
    depthdata: NDArray[tuple[int, int]],
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
    Iout = np.full((*depthdata.shape, len(light_angles)), np.nan)
    for i, light_angle in enumerate(light_angles):
        Iout[:, :, i] = calculate_surface(
            light_angle.vector, observer.vector, n1, n2, n3
        )
    return Iout
