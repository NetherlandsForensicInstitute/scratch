import numpy as np
from scipy.signal import convolve2d
from typing import Protocol

from utils.array_definitions import (
    NORMAL_VECTOR,
    IMAGE_2D_ARRAY,
    IMAGE_3_LAYER_STACK_ARRAY,
    IMAGE_3D_ARRAY,
)


def compute_surface_normals(
    depth_data: IMAGE_2D_ARRAY,
    x_dimension: float,
    y_dimension: float,
    kernel: tuple[tuple[int, int, int]] = ((0, 1j, 0), (1, 0, -1), (0, -1j, 0)),
) -> IMAGE_3_LAYER_STACK_ARRAY:
    """
    Compute per-pixel surface normals from a 2D depth map.

    Parameters
    ----------
    depth_data : IMAGE_2D_ARRAY
        2D array of depth values with shape (Height, Width).
    x_dimension : float
        Physical spacing between columns (Δx) in meters.
    y_dimension : float
        Physical spacing between rows (Δy) in meters.
    kernel : tuple of tuple of int, optional
        Complex convolution kernel used to approximate derivatives (default:
        ((0, 1j, 0), (1, 0, -1), (0, -1j, 0))).

    Returns
    -------
    IMAGE_3_STACK_ARRAY
        3D array of surface normals with shape (Height, Width, 3), where the last
        dimension corresponds to (nx, ny, nz).
    """
    factor_x = 1 / (2 * x_dimension)
    factor_y = 1 / (2 * y_dimension)

    z = convolve2d(depth_data, kernel, "same", fillvalue=np.nan)

    hx = z.real * factor_x
    hy = z.imag * factor_y

    norm = np.sqrt(hx * hx + hy * hy + 1)

    nx = -hx / norm
    ny = hy / norm
    nz = 1 / norm
    return np.stack([nx, ny, nz], axis=-1)


def calculate_lighting(
    light_vector: NORMAL_VECTOR,
    observer_vector: NORMAL_VECTOR,
    surface_normals: IMAGE_3_LAYER_STACK_ARRAY,
    specular_factor: float = 1.0,
    phong_exponent: int = 4,
) -> IMAGE_2D_ARRAY:
    """
    Compute per-pixel lighting intensity from a light source and surface normals.

    Uses a combination of Lambertian diffuse and Phong specular reflection.

    Parameters
    ----------
    light_vector : NORMAL_VECTOR
        3-element normalized vector pointing toward the light source.
    observer_vector : NORMAL_VECTOR
        3-element normalized vector pointing toward the observer/camera.
    surface_normals : IMAGE_3_STACK_ARRAY
        3D array of surface normals with shape (Height, Width, 3-layers(x,y,z).
    specular_factor : float, optional
        Weight of the specular component (default: 1.0).
    phong_exponent : int, optional
        Exponent controlling sharpness of specular highlights (default: 4).

    Returns
    -------
    IMAGE_2D_ARRAY
        2D array of combined lighting intensities in [0, 1] with shape (H, W).
    """
    h_vec = light_vector + observer_vector
    h_norm = np.linalg.norm(h_vec)
    h_vec = h_vec / h_norm

    nx, ny, nz = (
        surface_normals[..., 0],
        surface_normals[..., 1],
        surface_normals[..., 2],
    )

    diffuse = np.maximum(
        light_vector[0] * nx + light_vector[1] * ny + light_vector[2] * nz, 0
    )

    specular = np.maximum(h_vec[0] * nx + h_vec[1] * ny + h_vec[2] * nz, 0)
    specular = np.clip(specular, -1.0, 1.0)
    specular = np.maximum(np.cos(2 * np.arccos(specular)), 0) ** phong_exponent

    intensity = (diffuse + specular_factor * specular) / (1 + specular_factor)
    return intensity


class LightingCalculator(Protocol):
    def __call__(
        self,
        light_vector: NORMAL_VECTOR,
        observer_vector: NORMAL_VECTOR,
        surface_normals: IMAGE_3_LAYER_STACK_ARRAY,
        specular_factor: float = 1.0,
        phong_exponent: int = 4,
    ) -> IMAGE_2D_ARRAY: ...


def apply_multiple_lights(
    surface_normals: IMAGE_3_LAYER_STACK_ARRAY,
    light_vectors: tuple[NORMAL_VECTOR, ...],
    observer_vector: NORMAL_VECTOR,
    lighting_calculator: LightingCalculator = calculate_lighting,
) -> IMAGE_3D_ARRAY:
    """
    Apply multiple light sources to a surface and stack the resulting intensity maps.

    Parameters
    ----------
    surface_normals : IMAGE_3_STACK_ARRAY
        3D array of surface normals with shape (Height, Width, 3-layers(x,y,z)).
    light_vectors : tuple of NORMAL_VECTOR
        Tuple of normalized 3-element light direction vectors.
    observer_vector : NORMAL_VECTOR
        3-element normalized vector pointing toward the observer/camera.
    lighting_calculator : callable, optional
        Function that computes lighting for a single light source (default: `calculate_lighting`).

    Returns
    -------
    IMAGE_3D_ARRAY
        3D array of lighting intensities with shape (Height, Width, N), where N is the number of lights.
    """
    return np.stack(
        [
            lighting_calculator(light, observer_vector, surface_normals)
            for light in light_vectors
        ],
        axis=-1,
    )


def normalize_intensity_map(
    image_to_normalize: IMAGE_2D_ARRAY, max_val: float = 255, scale_min: float = 25
) -> IMAGE_2D_ARRAY:
    """
    Normalize a 2D intensity map to a specified range.

    Parameters
    ----------
    image_to_normalize : IMAGE_2D_ARRAY
        2D array of intensity values (Height, Width).
    max_val : float, optional
        Maximum value in the output image (default: 255).
    scale_min : float, optional
        Minimum value in the output image (default: 25).

    Returns
    -------
    IMAGE_2D_ARRAY
        Normalized 2D intensity map with values in [scale_min, max_val].
    """
    imin = np.nanmin(image_to_normalize, axis=(0, 1), keepdims=True)
    imax = np.nanmax(image_to_normalize, axis=(0, 1), keepdims=True)
    norm = (image_to_normalize - imin) / (imax - imin)
    return scale_min + (max_val - scale_min) * norm
