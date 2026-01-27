from collections.abc import Iterable
import numpy as np
from typing import Final
from returns.result import safe

from container_models.base import VectorField, UnitVector, FloatArray2D
from container_models.light_source import LightSource
from utils.logger import log_railway_function


SPECULAR_FACTOR: Final[float] = 1.0
PHONG_EXPONENT: Final[int] = 4


def _get_components(
    vector_field: VectorField,
) -> tuple[FloatArray2D, FloatArray2D, FloatArray2D]:
    return vector_field[..., 0], vector_field[..., 1], vector_field[..., 2]


def _compute_half_vector(
    light_vector: UnitVector, observer_vector: UnitVector
) -> UnitVector:
    """Compute and normalize the half-vector between light and observer directions."""
    h_vec = light_vector + observer_vector
    return h_vec / np.linalg.norm(h_vec)


def _compute_diffuse_lighting(
    surface_normals: VectorField, light_vector: UnitVector
) -> FloatArray2D:
    """Compute Lambertian diffuse reflection: max(N · L, 0)."""
    x_light, y_light, z_light = light_vector
    x_normal, y_normal, z_normal = _get_components(surface_normals)
    diffuse = np.maximum(
        x_light * x_normal + y_light * y_normal + z_light * z_normal,
        0,
    )
    return diffuse


def _compute_specular_lighting(
    half_vector: UnitVector, surface_normals: VectorField
) -> FloatArray2D:
    """
    Compute Phong specular reflection: max(cos(2*arccos(max(N · H, 0))), 0)^n.

    Uses the half-vector H between light and observer directions.
    """
    x_half_vector, y_half_vector, z_half_vector = half_vector
    x_normal, y_normal, z_normal = _get_components(surface_normals)

    specular = np.maximum(
        x_half_vector * x_normal + y_half_vector * y_normal + z_half_vector * z_normal,
        0,
    )
    specular = np.clip(specular, -1.0, 1.0)
    specular = np.maximum(np.cos(2 * np.arccos(specular)), 0) ** PHONG_EXPONENT

    return specular


def _combine_lighting_components(
    diffuse: FloatArray2D, specular: FloatArray2D
) -> FloatArray2D:
    """Combine diffuse and specular components with weighting factor."""
    combined = (diffuse + SPECULAR_FACTOR * specular) / (1 + SPECULAR_FACTOR)
    return combined


@log_railway_function("Calculating 2d maps per lighting source failed.")
def calculate_lighting(
    light: LightSource,
    observer: LightSource,
    surface_normals: VectorField,
) -> FloatArray2D:
    """
    Compute per-pixel lighting intensity from a light source and surface normals.

    Lighting is computed using Lambertian diffuse reflection combined with a
    Phong specular component.

    :param light: LightSource as a Normalized 3-element vector pointing toward the light source.
    :param observer: LightSource as Normalized 3-element vector pointing toward the observer/camera.
    :param surface_normals: Array data with 3D surface normals, shape: (Height, Width, 3).

    :returns: 2D array with the combined lighting intensities in ``[0, 1]``,
              with shape (Height, Width).
    """
    half_vector = _compute_half_vector(light.unit_vector, observer.unit_vector)
    diffuse = _compute_diffuse_lighting(surface_normals, light.unit_vector)
    specular = _compute_specular_lighting(half_vector, surface_normals)
    combined = _combine_lighting_components(diffuse, specular)
    return combined


@log_railway_function("Failed to apply lights")
@safe
def apply_multiple_lights(
    surface_normals: VectorField,
    light_sources: Iterable[LightSource],
    observer: LightSource,
) -> FloatArray2D:
    """
    Apply multiple directional light sources to a surface and combine them into
    a single intensity map.

    :param surface_normals: Array data with 3D surface normals, shape: (Height, Width, 3).
    :param light_sources: Iterable of LightSource objects representing directional lights.
    :param observer: LightSource representing the observer/camera position.

    :returns: 2D array with the combined lighting intensities with shape
              (Height, Width), where contributions from all lights are summed together.
    """
    lighting_results = [
        calculate_lighting(light, observer, surface_normals) for light in light_sources
    ]
    return np.nansum(
        np.stack([result.data for result in lighting_results], axis=-1),
        axis=2,
    )
