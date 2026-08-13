"""
Rotation and coordinate conventions shared by every stage of the search.

The rotation center is ``((width - 1) / 2, (height - 1) / 2)`` and the center of the input is
mapped onto the center of the (grown) output canvas.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from container_models.base import FloatArray2D


def pad_image_array(
    array: FloatArray2D, pad_width: int, pad_height: int, fill_value: float = np.nan
) -> FloatArray2D:
    """
    Pad a 2D array symmetrically with a constant fill value.

    :param array: Input 2D array of shape ``(height, width)``.
    :param pad_width: Number of columns to add on each side.
    :param pad_height: Number of rows to add on each side.
    :param fill_value: Constant written into the padded border.
    :returns: Padded array, same dtype as the input.
    """
    height, width = array.shape
    new_shape = height + 2 * pad_height, width + 2 * pad_width
    output = np.full(shape=new_shape, fill_value=fill_value, dtype=array.dtype)
    output[pad_height : pad_height + height, pad_width : pad_width + width] = array
    return output


def compute_rotated_shape(height: int, width: int, angle_deg: float) -> tuple[int, int]:
    """
    Output shape of rotate_image without performing the rotation.

    Calculates the axis-aligned bounding box of the rotated corner points.

    :param height: Input height in pixels.
    :param width: Input width in pixels.
    :param angle_deg: Rotation angle in degrees.
    :returns: ``(rotated_height, rotated_width)`` in pixels.
    """
    theta = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(theta)), abs(math.sin(theta))
    new_width = int(round((width - 1) * cos_a + (height - 1) * sin_a)) + 1
    new_height = int(round((width - 1) * sin_a + (height - 1) * cos_a)) + 1
    return new_height, new_width


def rotate_image(
    image: FloatArray2D, angle_deg: float, fill_value: float = np.nan
) -> FloatArray2D:
    """
    Rotate *image* by *angle_deg*, growing the canvas so no data is clipped.

    :param image: Input 2D array.
    :param angle_deg: Rotation angle in degrees.
    :param fill_value: Value written outside the rotated source rectangle.
    :returns: Float32 array of the computed rotated shape.
    """
    rotated_height, rotated_width = compute_rotated_shape(
        image.shape[0], image.shape[1], angle_deg
    )
    matrix = build_rotation_matrix(image.shape, angle_deg)
    return _warp(image, matrix, rotated_width, rotated_height, fill_value)


def crop_rotated_image(
    image: FloatArray2D,
    angle_deg: float,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    fill_value: float = np.nan,
) -> FloatArray2D:
    """
    Produce a crop of the rotated canvas without rotating the whole image.

    :param image: Source image.
    :param angle_deg: Rotation angle in degrees.
    :param left: Left edge of the desired crop, in rotated-canvas coordinates.
    :param top: Top edge of the desired crop, in rotated-canvas coordinates.
    :param crop_width: Crop width in pixels.
    :param crop_height: Crop height in pixels.
    :param fill_value: Value written outside the rotated source rectangle.
    :returns: Float32 array of shape ``(crop_height, crop_width)``.
    """
    matrix = build_rotation_matrix(image.shape, angle_deg, left=left, top=top)
    return _warp(image, matrix, crop_width, crop_height, fill_value)


def map_canvas_to_image(
    x: float,
    y: float,
    cell_shape: tuple[int, int],
    image_shape: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    """
    Map a matched window on the rotated canvas back to a cell center in the unrotated image.

    Inverse of map_image_to_canvas.

    :param x: Window left edge on the rotated canvas.
    :param y: Window top edge on the rotated canvas.
    :param cell_shape: ``(cell_height, cell_width)``.
    :param image_shape: ``(height, width)`` of the unrotated image.
    :param angle_deg: Rotation angle in degrees.
    :returns: The window's center in unrotated image coordinates.
    """
    cell_height, cell_width = cell_shape
    height, width = image_shape
    rotated_height, rotated_width = compute_rotated_shape(height, width, angle_deg)

    dx = x + cell_width / 2 - (rotated_width - 1) / 2
    dy = y + cell_height / 2 - (rotated_height - 1) / 2
    cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return (
        (width - 1) / 2 + cos_a * dx + sin_a * dy,
        (height - 1) / 2 - sin_a * dx + cos_a * dy,
    )


def map_image_to_canvas(
    center_x: float,
    center_y: float,
    cell_shape: tuple[int, int],
    image_shape: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    """
    Map a cell center in the unrotated image to a window's top-left corner on the rotated canvas.

    Inverse of map_canvas_to_image.

    :param center_x: Cell center x in unrotated image coordinates.
    :param center_y: Cell center y in unrotated image coordinates.
    :param cell_shape: ``(cell_height, cell_width)``.
    :param image_shape: ``(height, width)`` of the unrotated image.
    :param angle_deg: Rotation angle in degrees.
    :returns: The window's top-left corner on the rotated canvas.
    """
    cell_height, cell_width = cell_shape
    height, width = image_shape
    rotated_height, rotated_width = compute_rotated_shape(height, width, angle_deg)

    dx = center_x - (width - 1) / 2
    dy = center_y - (height - 1) / 2
    cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return (
        cos_a * dx - sin_a * dy + (rotated_width - 1) / 2 - cell_width / 2,
        sin_a * dx + cos_a * dy + (rotated_height - 1) / 2 - cell_height / 2,
    )


def map_coarse_to_full(coarse_value: float, cap_factor: float) -> float:
    """
    Map a coordinate in a downsampled image back onto the full-resolution grid.

    :param coarse_value: Coordinate in the downsampled image.
    :param cap_factor: Downsampling factor between the two grids.
    :returns: The matching full-resolution coordinate.
    """
    return coarse_value * cap_factor + (cap_factor - 1) / 2.0


def build_rotation_matrix(
    image_shape: tuple[int, ...], angle_deg: float, left: int = 0, top: int = 0
) -> np.ndarray:
    """
    Affine matrix rotating *image_shape* onto a canvas whose origin sits at ``(left, top)``.

    :param image_shape: ``(height, width)`` of the source image.
    :param angle_deg: Rotation angle in degrees.
    :param left: Canvas origin x, in rotated-canvas coordinates.
    :param top: Canvas origin y, in rotated-canvas coordinates.
    :returns: 2x3 affine transformation matrix.
    """
    height, width = image_shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotated_height, rotated_width = compute_rotated_shape(height, width, angle_deg)

    matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    # ``getRotationMatrix2D`` maps ``center`` onto itself; shift it onto the new canvas center.
    matrix[0, 2] += (rotated_width - 1) / 2.0 - center[0] - left
    matrix[1, 2] += (rotated_height - 1) / 2.0 - center[1] - top
    return matrix


def _warp(
    image: FloatArray2D,
    matrix: np.ndarray,
    width: int,
    height: int,
    fill_value: float,
) -> FloatArray2D:
    """
    Nearest-neighbor warp, so that no arithmetic is done across the NaN boundary.

    :param image: Source image.
    :param matrix: 2x3 affine transformation matrix.
    :param width: Output width in pixels.
    :param height: Output height in pixels.
    :param fill_value: Value written outside the warped source rectangle.
    :returns: Float32 array of shape ``(height, width)``.
    """
    return np.asarray(
        cv2.warpAffine(
            image.astype(np.float32, copy=False),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(fill_value),
        ),
        dtype=np.float32,
    )
