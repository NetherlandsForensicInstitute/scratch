import numpy as np
from typing import Protocol

from utils.array_definitions import (
    Vector3DArray,
    ScanMap2DArray,
    ScanVectorField2DArray,
    ScanTensor3DArray,
)


def compute_surface_normals(
    depth_data: ScanMap2DArray,
    x_dimension: float,
    y_dimension: float,
) -> ScanVectorField2DArray:
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

    Returns
    -------
    IMAGE_3_STACK_ARRAY
        3D array of surface normals with shape (Height, Width, 3), where the last
        dimension corresponds to (nx, ny, nz).
    """
    factor_x = 1 / (2 * x_dimension)
    factor_y = 1 / (2 * y_dimension)

    hx = (depth_data[:, :-2] - depth_data[:, 2:]) * factor_x
    hy = (depth_data[:-2, :] - depth_data[2:, :]) * factor_y
    hx = np.pad(hx, ((0, 0), (1, 1)), mode="constant", constant_values=np.nan)
    hy = np.pad(hy, ((1, 1), (0, 0)), mode="constant", constant_values=np.nan)
    norm = np.sqrt(hx * hx + hy * hy + 1)

    nx = hx / norm
    ny = -hy / norm
    nz = 1 / norm
    return np.stack([nx, ny, nz], axis=-1)


def calculate_lighting(
    light_vector: Vector3DArray,
    observer_vector: Vector3DArray,
    surface_normals: ScanVectorField2DArray,
    specular_factor: float = 1.0,
    phong_exponent: int = 4,
) -> ScanMap2DArray:
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
        2D array of combined lighting intensities in [0, 1] with shape (Height, Width).
    """
    h_vec = light_vector + observer_vector
    h_norm = np.linalg.norm(h_vec)
    h_vec /= h_norm

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
        light_vector: Vector3DArray,
        observer_vector: Vector3DArray,
        surface_normals: ScanVectorField2DArray,
        specular_factor: float = 1.0,
        phong_exponent: int = 4,
    ) -> ScanMap2DArray: ...


def apply_multiple_lights(
    surface_normals: ScanVectorField2DArray,
    light_vectors: tuple[Vector3DArray, ...],
    observer_vector: Vector3DArray,
    lighting_calculator: LightingCalculator = calculate_lighting,
) -> ScanTensor3DArray:
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
    image_to_normalize: ScanMap2DArray, max_val: float = 255, scale_min: float = 25
) -> ScanMap2DArray:
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
