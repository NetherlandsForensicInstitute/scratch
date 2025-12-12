from collections.abc import Iterable
import numpy as np
from typing import Final

from numpy.typing import ArrayLike, NDArray
from returns.pipeline import flow
from returns.result import safe

from container_models.light_source import LightSource
from container_models.scan_image import ScanImage
from utils.logger import log_railway_function


SPECULAR_FACTOR: Final[float] = 1.0
PHONG_EXPONENT: Final[int] = 4


def _get_normals(data: NDArray) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    return (data[..., 0], data[..., 1], data[..., 2])


def _compute_half_vector(scan_image: ScanImage) -> ScanImage:
    """Compute and normalize the half-vector between light and observer directions."""
    meta = scan_image.meta_data
    h_vec = meta["light_vector"] + meta.pop("observer_vector")
    return scan_image.model_copy(
        update={
            "meta_data": scan_image.meta_data
            | {
                "half_vector": h_vec / np.linalg.norm(h_vec),
            }
        },
    )


def _compute_diffuse_lighting(scan_image: ScanImage) -> ScanImage:
    """Compute Lambertian diffuse reflection: max(N · L, 0)."""
    meta = scan_image.meta_data
    x_light, y_light, z_light = meta.pop("light_vector")
    x_normal, y_normal, z_normal = _get_normals(scan_image.data)
    diffuse = np.maximum(
        x_light * x_normal + y_light * y_normal + z_light * z_normal,
        0,
    )
    return scan_image.model_copy(
        update={
            "meta_data": meta
            | {
                "diffuse": diffuse,
            }
        },
    )


def _compute_specular_lighting(scan_image: ScanImage) -> ScanImage:
    """
    Compute Phong specular reflection: max(cos(2*arccos(max(N · H, 0))), 0)^n.

    Uses the half-vector H between light and observer directions.
    """
    meta = scan_image.meta_data
    x_half_vector, y_half_vector, z_half_vector = meta.pop("half_vector")
    x_normal, y_normal, z_normal = _get_normals(scan_image.data)

    specular = np.maximum(
        x_half_vector * x_normal + y_half_vector * y_normal + z_half_vector * z_normal,
        0,
    )
    specular = np.clip(specular, -1.0, 1.0)
    specular = np.maximum(np.cos(2 * np.arccos(specular)), 0) ** PHONG_EXPONENT

    return scan_image.model_copy(
        update={
            "meta_data": meta
            | {
                "specular": specular,
            }
        },
    )


def _combine_lighting_components(scan_image: ScanImage) -> ScanImage:
    """Combine diffuse and specular components with weighting factor."""
    meta = scan_image.meta_data
    combined = (meta.pop("diffuse") + SPECULAR_FACTOR * meta.pop("specular")) / (
        1 + SPECULAR_FACTOR
    )

    return scan_image.model_copy(
        update=meta
        | {
            "data": combined,
        },
    )


@log_railway_function("Calculating 2d maps per lighting source failed.")
def calculate_lighting(
    light: LightSource,
    observer: LightSource,
    scan_image: ScanImage,
) -> ScanImage:
    """
    Compute per-pixel lighting intensity from a light source and surface normals.

    Lighting is computed using Lambertian diffuse reflection combined with a
    Phong specular component.

    :param light: LightSource as a Normalized 3-element vector pointing toward the light source.
    :param observer: LightSource as Normalized 3-element vector pointing toward the observer/camera.
    :param scan_image: ScanImage with 3D surface normals in data field (Height, Width, 3).

    :returns: ScanImage with 2D array of combined lighting intensities in ``[0, 1]``
              with shape (Height, Width).
    """
    scan_image = scan_image.model_copy(
        update={
            "meta_data": scan_image.meta_data
            | {
                "light_vector": light.unit_vector,
                "observer_vector": observer.unit_vector,
            }
        }
    )

    return flow(
        scan_image,
        _compute_half_vector,
        _compute_diffuse_lighting,
        _compute_specular_lighting,
        _combine_lighting_components,
    )


@log_railway_function("Failed to apply lights")
@safe
def apply_multiple_lights(
    scan_image: ScanImage,
    light_sources: Iterable[LightSource],
    observer: LightSource,
    scale_x: float,
    scale_y: float,
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
    lighting_results = [
        calculate_lighting(light, observer, scan_image) for light in light_sources
    ]

    return scan_image.model_copy(
        update={
            "data": np.nansum(
                np.stack([result.data for result in lighting_results], axis=-1),
                axis=2,
            ),
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
        deep=True,
    )
