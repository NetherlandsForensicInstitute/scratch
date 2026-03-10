from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=False)
class GridSearchParams:
    x: int = -1  # The x-coordinate to update
    y: int = -1  # The y-coordinate to update
    angle: float = 0.0  # The rotation angle to update
    score: float = float("-inf")  # The NCC score to update

    def update(self, x: int, y: int, angle: float, score: float):
        self.x = x
        self.y = y
        self.angle = angle
        self.score = score


@dataclass(frozen=True)
class GridCell:
    """Container class for storing generated grid cells."""

    center: tuple[int, int]  # Tuple of pixel coordinates (x, y)
    size: tuple[int, int]  # Tuple of cell size (width, height)
    cell_data: FloatArray2D  # Contains the sliced image data
    grid_search_params: GridSearchParams  # Params to optimize during search

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def center_x(self) -> int:
        return self.center[0]

    @property
    def center_y(self) -> int:
        return self.center[1]

    @property
    def top_left(self) -> tuple[int, int]:
        return self.center_x - self.width // 2, self.center_y - self.height // 2

    @property
    def fill_fraction(self) -> float:
        return float(1 - np.isnan(self.cell_data).sum() / self.cell_data.size)

    def fill_nans(self, fill_value: float):
        self.cell_data[np.isnan(self.cell_data)] = fill_value


def _convert_meters_to_pixels(value_to_convert: float, pixel_size: float) -> int:
    return int(round(value_to_convert / pixel_size))


def generate_grid(scan_image: ScanImage, params: ComparisonParams) -> list[GridCell]:
    # Create a dummy cell
    x, y = (scan_image.width // 2, scan_image.height // 2)
    width = _convert_meters_to_pixels(
        value_to_convert=params.cell_size[0], pixel_size=scan_image.scale_x
    )
    height = _convert_meters_to_pixels(
        value_to_convert=params.cell_size[1], pixel_size=scan_image.scale_y
    )
    dummy = GridCell(
        center=(x, y),
        size=(width, height),
        cell_data=scan_image.data[y : y + height, x : x + width],
        grid_search_params=GridSearchParams(),
    )
    return [dummy]
