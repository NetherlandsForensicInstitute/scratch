from dataclasses import dataclass

import numpy as np

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams
from conversion.surface_comparison.utils import (
    convert_meters_to_pixels,
    compute_fill_fraction,
)


@dataclass(frozen=False)
class GridSearchParams:
    """
    Mutable container for the best registration parameters found so far for one cell.

    All positional attributes are in pixel coordinates of the (rotated) comparison image.

    :param top_left_x: Top left x-coordinate of the best-matching comparison patch (pixels).
    :param top_left_y: Top left y-coordinate of the best-matching comparison patch (pixels).
    :param angle: Rotation angle at which the best score was found (degrees).
    :param score: Best normalized cross-correlation score found so far.
    """

    top_left_x: int = -1
    top_left_y: int = -1
    angle: float = 0.0
    score: float = float("-inf")

    def update(
        self, top_left_x: int, top_left_y: int, angle: float, score: float
    ) -> None:
        """Replace all fields with a new best result."""
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.angle = angle
        self.score = score


@dataclass(frozen=True)
class GridCell:
    """
    Immutable container for a single reference grid cell and its search state.

    All positional values are in pixel coordinates of the reference image.

    :param top_left: Top-left pixel coordinates (x, y) of the cell in the reference image.
    :param cell_data: 2D array of height data extracted from the reference image; may contain NaNs.
    :param grid_search_params: Mutable best-match state updated during coarse registration.
    """

    top_left: tuple[int, int]
    cell_data: FloatArray2D
    grid_search_params: GridSearchParams

    @property
    def width(self) -> int:
        return self.cell_data.shape[1]

    @property
    def height(self) -> int:
        return self.cell_data.shape[0]

    @property
    def center(self) -> tuple[float, float]:
        """Sub-pixel center derived from top-left and cell dimensions."""
        return self.top_left[0] + self.width / 2, self.top_left[1] + self.height / 2

    @property
    def fill_fraction(self) -> float:
        """Fraction of valid (non-NaN) pixels in the cell."""
        return compute_fill_fraction(self.cell_data)

    def fill_nans(self, fill_value: float) -> None:
        """Replace NaN entries in ``cell_data`` in-place with ``fill_value``."""
        self.cell_data[np.isnan(self.cell_data)] = fill_value

    def copy(self) -> "GridCell":
        return GridCell(
            top_left=self.top_left,
            cell_data=self.cell_data.copy(),
            grid_search_params=self.grid_search_params,
        )


def generate_grid(scan_image: ScanImage, params: ComparisonParams) -> list[GridCell]:
    """TODO: Implement full grid generation."""
    x, y = 0, 0
    width, height = convert_meters_to_pixels(
        values=params.cell_size, pixel_size=scan_image.scale_x
    )
    dummy = GridCell(
        top_left=(x, y),
        cell_data=scan_image.data[y : y + height, x : x + width],
        grid_search_params=GridSearchParams(),
    )
    return [dummy]


def extract_patch(
    scan_image: ScanImage,
    top_left: tuple[int, int],
    size: tuple[int, int],
    fill_value: float = np.nan,
) -> FloatArray2D:
    """
    Extract a patch from ``scan_image``, padding with ``fill_value`` where the patch extends outside the image boundary.

    :param scan_image: Source image to extract from.
    :param top_left: Top-left (x, y) pixel coordinate of the desired patch.
    :param size: Desired output patch size as (width, height) in pixels.
    :param fill_value: Value used for out-of-bounds padding.
    :returns: Array of shape (height, width) containing the extracted patch.
    """
    width, height = size
    x1, y1 = top_left
    x2, y2 = x1 + width, y1 + height

    # Compute global coordinates (= offset with respect to the reference image) for the patch
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(scan_image.width, x2)
    y2_clamped = min(scan_image.height, y2)

    output = np.full((height, width), fill_value=fill_value, dtype=np.float64)

    if x1_clamped >= x2_clamped or y1_clamped >= y2_clamped:
        return output  # Return NaN filled array

    # Compute local coordinates (= offset with respect to the patch) for the sliced data
    local_x = x1_clamped - x1
    local_y = y1_clamped - y1
    valid_width = x2_clamped - x1_clamped
    valid_height = y2_clamped - y1_clamped

    output[local_y : local_y + valid_height, local_x : local_x + valid_width] = (
        scan_image.data[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    )

    return output
