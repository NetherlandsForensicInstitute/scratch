from functools import partial
import numpy as np
from typing import Final, NamedTuple

from numpy.typing import NDArray
from returns.pipeline import flow
from returns.result import safe

from container_models.scan_image import ScanImage
from container_models.surface_normals import SurfaceNormals
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
_PAD_X_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (1, 1),
)  # Pad left and right (columns)
_PAD_Y_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (0, 0),
)  # Pad top and bottom (rows)


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


def _normalize_to_surface_normals(gradients: GradientComponents) -> SurfaceNormals:
    """Normalize gradient components to unit surface normal vectors."""
    x, y, magnitude = gradients
    if magnitude is None:
        raise ValueError
    return SurfaceNormals(
        x_normal_vector=x / magnitude,
        y_normal_vector=-y / magnitude,
        z_normal_vector=1 / magnitude,
    )


@log_railway_function(
    failure_message="Failed to compute surface normals from depth data",
    success_message="Successfully computed surface normal components",
)
@safe
def compute_surface_normals(scan_image: ScanImage) -> SurfaceNormals:
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
        PhysicalSpacing(scan_image.scale_x, scan_image.scale_y),
        _compute_central_diff_scales,
        partial(_compute_depth_gradients, depth_data=scan_image.data),
        _add_normal_magnitude,
        _normalize_to_surface_normals,
    )


@log_railway_function(
    failure_message="Failed to normalize 2D intensity map",
)
@safe
def normalize_2d_array(
    image_to_normalize: ScanImage,
    scale_max: float = 255,
    scale_min: float = 25,
) -> ScanImage:
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
    imin = np.nanmin(image_to_normalize.data)
    imax = np.nanmax(image_to_normalize.data)
    norm = (image_to_normalize.data - imin) / (imax - imin)
    return image_to_normalize.model_copy(
        update={"data": scale_min + (scale_max - scale_min) * norm}
    )
