from collections.abc import Iterable
import numpy as np
from typing import Final, NamedTuple

from returns.pipeline import flow
from returns.result import safe

from container_models.light_source import LightSource
from container_models.scan_image import ScanImage
from container_models.surface_normals import SurfaceNormals
from container_models.base import UnitVector3DArray, ScanMap2DArray
from utils.logger import log_railway_function


SPECULAR_FACTOR: Final[float] = 1.0
PHONG_EXPONENT: Final[int] = 4


class LightingComponents(NamedTuple):
    """Container for lighting calculation components."""

    light_vector: LightSource
    observer_vector: LightSource
    surface_normals: SurfaceNormals
    half_vector: UnitVector3DArray | None = None
    diffuse: ScanMap2DArray | None = None
    specular: ScanMap2DArray | None = None


def _compute_half_vector(components: LightingComponents) -> LightingComponents:
    """Compute and normalize the half-vector between light and observer directions."""
    h_vec = components.light_vector.unit_vector + components.observer_vector.unit_vector
    return components._replace(half_vector=h_vec / np.linalg.norm(h_vec))


def _compute_diffuse_lighting(components: LightingComponents) -> LightingComponents:
    """Compute Lambertian diffuse reflection: max(N · L, 0)."""
    x_light, y_light, z_light = components.light_vector.unit_vector
    return components._replace(
        diffuse=np.maximum(
            x_light * components.surface_normals.x_normal_vector
            + y_light * components.surface_normals.y_normal_vector
            + z_light * components.surface_normals.z_normal_vector,
            0,
        )
    )


def _compute_specular_lighting(components: LightingComponents) -> LightingComponents:
    """
    Compute Phong specular reflection: max(cos(2*arccos(max(N · H, 0))), 0)^n.

    Uses the half-vector H between light and observer directions.
    """

    if components.half_vector is None:
        raise AttributeError

    x_half_vector, y_half_vector, z_half_vector = components.half_vector

    specular = np.maximum(
        x_half_vector * components.surface_normals.x_normal_vector
        + y_half_vector * components.surface_normals.y_normal_vector
        + z_half_vector * components.surface_normals.z_normal_vector,
        0,
    )
    specular = np.clip(specular, -1.0, 1.0)
    specular = np.maximum(np.cos(2 * np.arccos(specular)), 0) ** PHONG_EXPONENT

    return components._replace(specular=specular)


def _combine_lighting_components(
    components: LightingComponents,
) -> ScanMap2DArray:
    """Combine diffuse and specular components with weighting factor."""

    if components.diffuse is None or components.specular is None:
        raise AttributeError

    return (components.diffuse + SPECULAR_FACTOR * components.specular) / (
        1 + SPECULAR_FACTOR
    )


@log_railway_function("Calculating 2d maps per lighting source failed.")
def calculate_lighting(
    light_vector: LightSource,
    observer_vector: LightSource,
    surface_normals: SurfaceNormals,
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
    return flow(
        LightingComponents(light_vector, observer_vector, surface_normals),
        _compute_half_vector,
        _compute_diffuse_lighting,
        _compute_specular_lighting,
        _combine_lighting_components,
    )


@log_railway_function("Failed to apply lights")
@safe
def apply_multiple_lights(
    surface_normals: SurfaceNormals,
    light_sources: Iterable[LightSource],
    observer: LightSource,
    scale_x: float,
    scale_y: float,
) -> ScanImage:
    """
    Apply multiple directional light sources to a surface and combine them into
    a single intensity map.

    :param surface_normals: 3D array of surface normals with shape (Height, Width, 3).
    :param light_vectors: Tuple of normalized 3-element light direction vectors.
    :param observer_vector: Normalized 3-element vector pointing toward the observer.
    :param lighting_calculator: Function used to compute lighting for a single light
                                source. Default is :func:`calculate_lighting`.

    :returns: ScanImage with 2D array of combined lighting intensities with shape
              (Height, Width), where contributions from all lights are summed together.
    """

    return ScanImage(
        data=np.nansum(
            np.stack(
                [
                    calculate_lighting(light, observer, surface_normals)
                    for light in light_sources
                ],
                axis=-1,
            ),
            axis=2,
        ),
        scale_x=scale_x,
        scale_y=scale_y,
    )
