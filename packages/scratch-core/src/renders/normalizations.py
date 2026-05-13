from typing import Final

import numpy as np

from container_models.base import FloatArray2D, VectorField
from container_models.scan_image import ScanImage

# Padding configurations for gradient arrays to maintain original dimensions
_PAD_X_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (1, 1),
)  # Pad left and right (columns)
_PAD_Y_GRADIENT: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (0, 0),
)  # Pad top and bottom (rows)


def _pad_gradient(
    unpadded_gradient: FloatArray2D, pad_width: tuple[tuple[int, int], tuple[int, int]]
) -> FloatArray2D:
    """Pad a gradient array with NaN values at the borders."""
    return np.pad(unpadded_gradient, pad_width, mode="constant", constant_values=np.nan)


def _compute_depth_gradients(
    scan_image: ScanImage,
) -> tuple[FloatArray2D, FloatArray2D]:
    """Compute depth gradients (∂z/∂x, ∂z/∂y) using central differences."""

    def _compute(value: float) -> float:
        """Compute scaling factors for central difference approximation: 1/(2*spacing)."""
        return 1 / (2 * value)

    gradient_x = _pad_gradient(
        (scan_image.data[:, :-2] - scan_image.data[:, 2:])
        * _compute(scan_image.scale_x),
        _PAD_X_GRADIENT,
    )
    gradient_y = _pad_gradient(
        (scan_image.data[:-2, :] - scan_image.data[2:, :])
        * _compute(scan_image.scale_y),
        _PAD_Y_GRADIENT,
    )
    return gradient_x, gradient_y


def _compute_magnitude(
    gradient_x: FloatArray2D, gradient_y: FloatArray2D
) -> FloatArray2D:
    """Compute the vector magnitude for gradient components."""
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + 1)
    return magnitude


def _normalize_to_surface_normals(
    gradient_x: FloatArray2D, gradient_y: FloatArray2D, magnitude: FloatArray2D
) -> VectorField:
    """Normalize gradient components to unit surface normal vectors."""
    return np.stack(
        [
            gradient_x / magnitude,
            -gradient_y / magnitude,
            1 / magnitude,
        ],
        axis=-1,
    )


def compute_surface_normals(scan_image: ScanImage) -> VectorField:
    """
    Compute per-pixel surface normals from a 2D depth map.

    The gradients in both x and y directions are estimated using central differences,
    and the resulting normal vectors are normalized per pixel.
    The border are padded with NaN values to keep the same size as the input data.

    :param scan_image: A ScanImage where the mutation/ calculation is being made on.

    :returns: 3D array of surface normals with shape (Height, Width, 3), where the
              last dimension corresponds to (nx, ny, nz).
    """

    gradient_x, gradient_y = _compute_depth_gradients(scan_image)
    magnitude = _compute_magnitude(gradient_x, gradient_y)
    surface_normals = _normalize_to_surface_normals(gradient_x, gradient_y, magnitude)
    return surface_normals
