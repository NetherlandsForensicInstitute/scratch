from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel
from container_models.base import DepthData, VectorField
from container_models.light_source import LightSource
from image_tasks.types.base import FloatArray2D, UnitVector
from image_tasks.factory import create_image_task
from image_tasks.types.scan_image import Point, ScanImage
from returns.result import safe
import numpy as np


# Padding configurations for gradient arrays to maintain original dimensions
_PAD_X_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (1, 1),
)  # Pad left and right (columns)
_PAD_Y_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (0, 0),
)  # Pad top and bottom (rows)
MAX_SCALE: Final[int] = 255
MIN_SCALE: Final[int] = 25
SPECULAR_FACTOR: Final[float] = 1.0
PHONG_EXPONENT: Final[int] = 4


def _pad_gradient(
    unpadded_gradient: FloatArray2D, pad_width: tuple[tuple[int, int], tuple[int, int]]
) -> FloatArray2D:
    """Pad a gradient array with NaN values at the borders."""
    return np.pad(unpadded_gradient, pad_width, mode="constant", constant_values=np.nan)


def _normalize_to_surface_normals(data: DepthData, scale: Point[float]) -> VectorField:
    """Normalize gradient components to unit surface normal vectors."""
    gradient_x = _pad_gradient(
        (data[:, :-2] - data[:, 2:]) * scale.x,
        _PAD_X_GRADIENT,
    )
    gradient_y = _pad_gradient(
        (data[:-2, :] - data[2:, :]) * scale.y,
        _PAD_Y_GRADIENT,
    )
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + 1)
    return np.stack(
        [gradient_x / magnitude, -gradient_y / magnitude, 1 / magnitude], axis=-1
    )


def _normalize_2d_array(data: VectorField) -> DepthData:
    """
    Normalize a 2D intensity map to a specified output range.

    The normalization is done by the steps:
    1. apply min-max normalization to grayscale data
    2. stretch / scale the normalized data from the unit range to a specified output range

    :param image_to_normalize: 2D array of input intensity values.

    :returns: Normalized 2D intensity map with values in ``[scale_min, max_val]``.
    """
    imin = np.nanmin(data)
    imax = np.nanmax(data)
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (data - imin) / (imax - imin)


class MultipleLightParam(BaseModel):
    light_sources: Iterable[LightSource]
    observer: LightSource
    scale: Point[float]


def _half_vector(light: UnitVector, observer: UnitVector) -> UnitVector:
    """Compute and normalize the half-vector between light and observer directions."""
    return (vector := light + observer) / np.linalg.norm(vector)


def _diffuse_light(data: DepthData, light: UnitVector) -> DepthData:
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


def _combine_lighting_components(
    data: DepthData, light: UnitVector, observer: UnitVector
) -> DepthData:
    """Combine diffuse and specular components with weighting factor."""
    return (
        _diffuse_light(data, light)
        + SPECULAR_FACTOR * _specular_light(data, _half_vector(light, observer))
    ) / (1 + SPECULAR_FACTOR)


@safe
def _apply_multiple_lights(
    scan_image: ScanImage,
    *,
    light_sources: Iterable[LightSource],
    observer: LightSource,
    scale: Point[float],
) -> ScanImage:
    """
    Apply multiple directional light sources to a surface and combine them into
    a single intensity map.

    :param scan_image: ScanImage with 3D surface normals in data field (Height, Width, 3).
    :param light_sources: Iterable of LightSource objects representing directional lights.
    :param observer: LightSource representing the observer/camera position.
    :param scale_x: Physical spacing in x direction (meters).
    :param scale_y: Physical spacing in y direction (meters).

    :returns: ScanImage with 2D array of combined lighting intensities with shape
              (Height, Width), where contributions from all lights are summed together.
    """
    normalize_data = _normalize_to_surface_normals(
        scan_image.data, scan_image.meta_data.central_diff_scales
    )
    data = np.nansum(
        np.stack(
            [
                _combine_lighting_components(
                    normalize_data, light.unit_vector, observer.unit_vector
                )
                for light in light_sources
            ],
            axis=-1,
        ),
        axis=2,
    )
    scan_image.data = _normalize_2d_array(data)
    scan_image.meta_data.scale = scale
    return scan_image


apply_multiple_lights = create_image_task(
    _apply_multiple_lights,
    params_model=MultipleLightParam,
    failure_msg="Failed to apply lights to image",
    success_msg="Successfully applied lights to image",
)
