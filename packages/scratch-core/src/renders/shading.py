import numpy as np
from typing import Final
from container_models.base import DepthData, ImageData, UnitVector


SPECULAR_FACTOR: Final[float] = 1.0
PHONG_EXPONENT: Final[int] = 4


def _half_vector(light: UnitVector, observer: UnitVector) -> UnitVector:
    """Compute and normalize the half-vector between light and observer directions."""
    return (vector := light + observer) / np.linalg.norm(vector)


def _diffuse_light(data: ImageData, light: UnitVector) -> DepthData:
    """Compute Lambertian diffuse reflection: max(N · L, 0)."""
    x_light, y_light, z_light = light
    x_normal, y_normal, z_normal = data[..., 0], data[..., 1], data[..., 2]
    return np.maximum(
        x_light * x_normal + y_light * y_normal + z_light * z_normal,
        0,
    )


def _specular_light(data: DepthData, half_vector: UnitVector) -> DepthData:
    """
    Compute Phong specular reflection: max(cos(2*arccos(max(N · H, 0))), 0)^n.

    Uses the half-vector H between light and observer directions.
    """
    specular = np.clip(_diffuse_light(data, half_vector), -1.0, 1.0)
    return np.maximum(np.cos(2 * np.arccos(specular)), 0) ** PHONG_EXPONENT


def combine_lighting_components(
    data: DepthData, light: UnitVector, observer: UnitVector
) -> DepthData:
    """Combine diffuse and specular components with weighting factor."""
    return (
        _diffuse_light(data, light)
        + SPECULAR_FACTOR * _specular_light(data, _half_vector(light, observer))
    ) / (1 + SPECULAR_FACTOR)
