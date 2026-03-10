from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams

from dataclasses import dataclass
import numpy as np

from conversion.surface_comparison.utils import convert_meters_to_pixels


@dataclass(frozen=True)
class GridCell:
    """Container class for storing generated grid cells."""

    center: tuple[
        int, int
    ]  # Tuple of pixel coordinates (x, y) corresponding to the reference image.
    cell_data: FloatArray2D  # Contains the sliced image data

    @property
    def width(self) -> int:
        return self.cell_data.shape[1]

    @property
    def height(self) -> int:
        return self.cell_data.shape[0]

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


def generate_grid(scan_image: ScanImage, params: ComparisonParams) -> list[GridCell]:
    """TODO: Implement function."""

    # Create a dummy cell
    x, y = (scan_image.width // 2, scan_image.height // 2)
    width = convert_meters_to_pixels(
        value_to_convert=params.cell_size[0], pixel_size=scan_image.scale_x
    )
    height = convert_meters_to_pixels(
        value_to_convert=params.cell_size[1], pixel_size=scan_image.scale_y
    )
    dummy = GridCell(
        center=(x, y), cell_data=scan_image.data[y : y + height, x : x + width]
    )
    return [dummy]
