import numpy as np
from typing import Protocol

from utils.array_definitions import (
    UnitVector3DArray,
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

    The gradients in both x and y directions are estimated using central differences,
    and the resulting normal vectors are normalized per pixel.
    The border are padded with NaN values to keep the same size as the input data.

    :param depth_data: 2D array of depth values with shape (Height, Width).
    :param x_dimension: Physical spacing between columns (Δx) in meters.
    :param y_dimension: Physical spacing between rows (Δy) in meters.

    :returns: 3D array of surface normals with shape (Height, Width, 3), where the
              last dimension corresponds to (nx, ny, nz).
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
    light_vector: UnitVector3DArray,
    observer_vector: UnitVector3DArray,
    surface_normals: ScanVectorField2DArray,
    specular_factor: float = 1.0,
    phong_exponent: int = 4,
) -> ScanMap2DArray:
    """
    Compute per-pixel lighting intensity from a light source and surface normals.

    Lighting is computed using Lambertian diffuse reflection combined with a
    Phong specular component.

    :param light_vector: Normalized 3-element vector pointing toward the light source.
    :param observer_vector: Normalized 3-element vector pointing toward the observer/camera.
    :param surface_normals: 3D array of surface normals with shape (Height, Width, 3).
    :param specular_factor: Weight of the specular component. Default is ``1.0``.
    :param phong_exponent: Exponent controlling the sharpness of specular highlights.
                           Default is ``4``.

    :returns: 2D array of combined lighting intensities in ``[0, 1]`` with shape
              (Height, Width).
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

    return (diffuse + specular_factor * specular) / (1 + specular_factor)


class LightingCalculator(Protocol):
    def __call__(
        self,
        light_vector: UnitVector3DArray,
        observer_vector: UnitVector3DArray,
        surface_normals: ScanVectorField2DArray,
        specular_factor: float = 1.0,
        phong_exponent: int = 4,
    ) -> ScanMap2DArray: ...


def apply_multiple_lights(
    surface_normals: ScanVectorField2DArray,
    light_vectors: tuple[UnitVector3DArray, ...],
    observer_vector: UnitVector3DArray,
    lighting_calculator: LightingCalculator = calculate_lighting,
) -> ScanTensor3DArray:
    """
    Apply multiple directional light sources to a surface and stack the
    resulting intensity maps.

    :param surface_normals: 3D array of surface normals with shape (Height, Width, 3).
    :param light_vectors: Tuple of normalized 3-element light direction vectors.
    :param observer_vector: Normalized 3-element vector pointing toward the observer.
    :param lighting_calculator: Function used to compute lighting for a single light
                                source. Default is :func:`calculate_lighting`.

    :returns: 3D array of lighting intensities with shape (Height, Width, N), where
              N is the number of lights.
    """
    return np.stack(
        [
            lighting_calculator(light, observer_vector, surface_normals)
            for light in light_vectors
        ],
        axis=-1,
    )


def normalize_2d_array(
    image_to_normalize: ScanMap2DArray,
    scale_max: float = 255,
    scale_min: float = 25,
) -> ScanMap2DArray:
    """
    Normalize a 2D intensity map to a specified output range.

    The normalization is done by the steps:
    1. apply min-max normalization to grayscale data
    2. stretch / scale the normalized data from the unit range to a specified output range

    :param image_to_normalize: 2D array of input intensity values.
    :param scale_max: Maximum output intensity value. Default is ``255``.
    :param scale_min: Minimum output intensity value. Default is ``25``.

    :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
    """
    imin = np.nanmin(image_to_normalize)
    imax = np.nanmax(image_to_normalize)
    norm = (image_to_normalize - imin) / (imax - imin)
    return scale_min + (scale_max - scale_min) * norm
