from collections.abc import Callable
from PIL.Image import Image, fromarray
import numpy as np
from typing import NamedTuple, Protocol

from numpy.typing import NDArray
from returns.pipeline import flow
from returns.result import safe

from utils.array_definitions import (
    UnitVector3DArray,
    ScanMap2DArray,
    ScanVectorField2DArray,
    ScanTensor3DArray,
    ScanMapRGBA,
)
from utils.logger import log_railway_function


class GradientComponents(NamedTuple):
    """Container for gradient components with optional magnitude."""

    x: NDArray
    y: NDArray
    magnitude: NDArray | None = None


class PhysicalSpacing(NamedTuple):
    """Physical spacing between samples in x and y directions."""

    x: float
    y: float


# Padding configurations for gradient arrays to maintain original dimensions
_PAD_X_GRADIENT = ((0, 0), (1, 1))  # Pad left and right (columns)
_PAD_Y_GRADIENT = ((1, 1), (0, 0))  # Pad top and bottom (rows)


def _compute_central_diff_scales(
    spacing: PhysicalSpacing,
) -> PhysicalSpacing:
    """Compute scaling factors for central difference approximation: 1/(2*spacing)."""
    return PhysicalSpacing(*(1 / (2 * value) for value in spacing))


def _pad_gradient(
    unpadded_gradient: NDArray, pad_width: tuple[tuple[int, int], tuple[int, int]]
) -> NDArray:
    """Pad a gradient array with NaN values at the borders."""
    return np.pad(unpadded_gradient, pad_width, mode="constant", constant_values=np.nan)


def _compute_depth_gradients(
    scales: PhysicalSpacing, depth_data: NDArray
) -> GradientComponents:
    """Compute depth gradients (∂z/∂x, ∂z/∂y) using central differences."""
    return GradientComponents(
        x=_pad_gradient(
            (depth_data[:, :-2] - depth_data[:, 2:]) * scales.x,
            _PAD_X_GRADIENT,
        ),
        y=_pad_gradient(
            (depth_data[:-2, :] - depth_data[2:, :]) * scales.y,
            _PAD_Y_GRADIENT,
        ),
    )


def _add_normal_magnitude(gradients: GradientComponents) -> GradientComponents:
    """Compute and attach the normal vector magnitude to gradient components."""
    magnitude = np.sqrt(gradients.x**2 + gradients.y**2 + 1)
    return GradientComponents(gradients.x, gradients.y, magnitude)


def _normalize_to_surface_normals(gradients: GradientComponents) -> NDArray:
    """Normalize gradient components to unit surface normal vectors."""
    x, y, magnitude = gradients
    assert magnitude  # for type checker
    return np.stack([x / magnitude, -y / magnitude, 1 / magnitude], axis=-1)


@log_railway_function(
    failure_message="Failed to compute surface normals from depth data",
    success_message="Successfully computed surface normal components",
)
@safe
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
    return flow(
        PhysicalSpacing(x_dimension, y_dimension),
        _compute_central_diff_scales,
        lambda scales: _compute_depth_gradients(scales, depth_data),
        _add_normal_magnitude,
        _normalize_to_surface_normals,
    )


class LightingComponents(NamedTuple):
    """Container for lighting calculation components."""

    light_vector: UnitVector3DArray
    observer_vector: UnitVector3DArray
    surface_normals: ScanVectorField2DArray
    half_vector: UnitVector3DArray | None = None
    normal_components: tuple[NDArray, NDArray, NDArray] | None = None
    diffuse: ScanMap2DArray | None = None
    specular: ScanMap2DArray | None = None


def _compute_half_vector(components: LightingComponents) -> LightingComponents:
    """Compute and normalize the half-vector between light and observer directions."""
    h_vec = components.light_vector + components.observer_vector
    h_vec = h_vec / np.linalg.norm(h_vec)
    return components._replace(half_vector=h_vec)


def _extract_normal_components(components: LightingComponents) -> LightingComponents:
    """Extract individual x, y, z components from surface normal field."""
    nx, ny, nz = np.moveaxis(components.surface_normals, -1, 0)
    return components._replace(normal_components=(nx, ny, nz))


def _compute_diffuse_lighting(components: LightingComponents) -> LightingComponents:
    """Compute Lambertian diffuse reflection: max(N · L, 0)."""
    nx, ny, nz = components.normal_components  # type: ignore
    light = components.light_vector
    diffuse = np.maximum(light[0] * nx + light[1] * ny + light[2] * nz, 0)
    return components._replace(diffuse=diffuse)


def _compute_specular_lighting(
    phong_exponent: int,
) -> Callable[[LightingComponents], LightingComponents]:
    """
    Compute Phong specular reflection: max(cos(2*arccos(max(N · H, 0))), 0)^n.

    Uses the half-vector H between light and observer directions.
    """

    def _compute(components: LightingComponents) -> LightingComponents:
        nx, ny, nz = components.normal_components  # type: ignore
        h_vec = components.half_vector  # type: ignore

        specular = np.maximum(h_vec[0] * nx + h_vec[1] * ny + h_vec[2] * nz, 0)
        specular = np.clip(specular, -1.0, 1.0)
        specular = np.maximum(np.cos(2 * np.arccos(specular)), 0) ** phong_exponent

        return components._replace(specular=specular)

    return _compute


def _combine_lighting_components(
    specular_factor: float,
) -> Callable[[LightingComponents], ScanMap2DArray]:
    """Combine diffuse and specular components with weighting factor."""

    def _combine(components: LightingComponents) -> ScanMap2DArray:
        diffuse = components.diffuse  # type: ignore
        specular = components.specular  # type: ignore
        return (diffuse + specular_factor * specular) / (1 + specular_factor)

    return _combine


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
    return flow(
        LightingComponents(light_vector, observer_vector, surface_normals),
        _compute_half_vector,
        _extract_normal_components,
        _compute_diffuse_lighting,
        _compute_specular_lighting(phong_exponent),
        _combine_lighting_components(specular_factor),
    )


class LightingCalculator(Protocol):
    def __call__(
        self,
        light_vector: UnitVector3DArray,
        observer_vector: UnitVector3DArray,
        surface_normals: ScanVectorField2DArray,
        specular_factor: float = 1.0,
        phong_exponent: int = 4,
    ) -> ScanMap2DArray: ...


@safe
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


def _grayscale_to_rgba(scan_data: ScanMap2DArray) -> ScanMapRGBA:
    """
    Convert a 2D grayscale array to an 8-bit RGBA array.

    The grayscale pixel values are assumed to be floating point values in the [0, 255] interval.
    NaN values will be converted to black pixels with 100% transparency.

    :param image: The grayscale image data to be converted to an 8-bit RGBA image.
    :returns: Array with the image data in 8-bit RGBA format.
    """
    rgba = np.empty(shape=(*scan_data.shape, 4), dtype=np.uint8)
    rgba[..., :-1] = np.expand_dims(scan_data.astype(np.uint8), axis=-1)
    rgba[..., -1] = ~np.isnan(scan_data) * 255
    return rgba


def scan_to_image(scan_data: ScanMap2DArray) -> Image:
    return fromarray(_grayscale_to_rgba(scan_data=scan_data))
