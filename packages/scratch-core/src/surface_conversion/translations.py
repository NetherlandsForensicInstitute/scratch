import numpy as np
from numpy._typing import NDArray
from scipy.signal import convolve2d

from surface_conversion.data_formats import DepthMap, SurfaceNormals, LightVector
from surface_conversion.schemas import LightAngle


def compute_surface_normals(
    depth_data: DepthMap,
    kernel: tuple[tuple[int, int, int]] = ((0, 1j, 0), (1, 0, -1), (0, -1j, 0)),
) -> SurfaceNormals:
    """
    Compute surface-normal components (n1, n2, n3) from a 2D depth map.

    Parameters
    ----------
    depth_data : DepthMap
        array representing a 2D image.
    xdim : float
        Physical spacing between columns in meters (Δx).
    ydim : float
        Physical spacing between rows in meters (Δy).
    kernel : NDArray
        the kernel used to convolve the diff of the neighboring cells

    """
    factor_x = 1 / (2 * depth_data.xdim)
    factor_y = 1 / (2 * depth_data.ydim)

    z = convolve2d(depth_data.data, kernel, "same", fillvalue=np.nan)

    hx = z.real * factor_x
    hy = z.imag * factor_y

    norm = np.sqrt(hx * hx + hy * hy + 1)

    return SurfaceNormals(
        nx=-hx / norm,
        ny=hy / norm,
        nz=1 / norm,
    )


def calculate_lighting(
    light_vector: LightVector,
    observer_vector: LightVector,
    surface_normals: SurfaceNormals,
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
    surface_normals : SurfaceNormals
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

    diffuse = (
        light_vector[0] * surface_normals.nx
        + light_vector[1] * surface_normals.ny
        + light_vector[2] * surface_normals.nz
    )
    diffuse = np.maximum(diffuse, 0)

    specular = (
        h[0] * surface_normals.nx
        + h[1] * surface_normals.ny
        + h[2] * surface_normals.nz
    )
    specular = np.maximum(specular, 0)
    specular = np.cos(2 * np.arccos(specular))
    specular = np.maximum(specular, 0) ** phong_exponent

    intensity = (diffuse + specular_factor * specular) / (1 + specular_factor)
    return intensity


def apply_multiple_lights(
    surface_normals: SurfaceNormals,
    light_angles: tuple[LightAngle],
    observer: LightAngle = LightAngle(azimuth=0, elevation=90),
    lighting_calculator=calculate_lighting,
):
    """

    Parameters
    ----------
    surface_normals
    light_angles
    observer
    lighting_calculator

    Returns
    -------

    """
    return np.stack(
        [
            lighting_calculator(light_angle.vector, observer.vector, surface_normals)
            for light_angle in light_angles
        ],
        axis=-1,
    )


def pre_refactor_logic(
    depthdata: DepthMap,
    light_angles: tuple[LightAngle] = (
        LightAngle(azimuth=90, elevation=45),
        LightAngle(azimuth=180, elevation=45),
    ),
    famb=25,
):
    surface_normals = compute_surface_normals(depth_data=depthdata)

    # Calculate intensity of surface for each light source
    Iout = apply_multiple_lights(surface_normals, light_angles)

    # Calculate total intensity of surface
    Iout = np.nansum(Iout, axis=2)

    # Normalize between [0,1]
    Imin = np.nanmin(Iout)
    Imax = np.nanmax(Iout)
    Iout = (Iout - Imin) / (Imax - Imin)

    # Add ambient component and scale [0,1]->[0,255]
    Iout = famb + (255 - famb) * Iout

    return Iout
