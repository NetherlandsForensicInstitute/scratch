from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams

from dataclasses import dataclass
import numpy as np

from conversion.surface_comparison.utils import convert_meters_to_pixels


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
    """
    Container class for storing generated grid cells.

    All the values of the attributes and properties are in pixel units.

    :param top_left: Tuple containing the top-left pixel coordinates (x, y) corresponding to the reference image.
    :param cell_data: 2D array containing the sliced image data from the reference image.
    :param grid_search_params: Params to optimize during search
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
        return self.top_left[0] + self.width / 2, self.top_left[1] + self.height / 2

    @property
    def fill_fraction(self) -> float:
        return float(np.count_nonzero(~np.isnan(self.cell_data)) / self.cell_data.size)

    def fill_nans(self, fill_value: float):
        self.cell_data[np.isnan(self.cell_data)] = fill_value


def generate_grid(scan_image: ScanImage, params: ComparisonParams) -> list[GridCell]:
    """TODO: Implement function."""

    # Create a dummy cell
    x, y = 0, 0  # Top-left coordinates of cell in reference image
    pixel_size = scan_image.scale_x  # Assumes isotropic image
    width, height = convert_meters_to_pixels(
        values=params.cell_size, pixel_size=pixel_size
    )
    dummy = GridCell(
        top_left=(x, y),
        cell_data=scan_image.data[y : y + height, x : x + width],
        grid_search_params=GridSearchParams(),
    )
    return [dummy]


def extract_patch(
    scan_image: ScanImage,
    coordinates: tuple[int, int],  # Top-left (x, y) coordinates
    size: tuple[int, int],  # Usually a square (width, height)
    fill_value: float = np.nan,
) -> FloatArray2D:
    """TODO: Implement function."""
    # Compute global coordinates (of the patch in the image)
    x1, y1 = coordinates
    width, height = size
    x2, y2 = x1 + width, y1 + height
    x1, x2 = max(0, x1), min(scan_image.width, x2)
    y1, y2 = max(0, y1), min(scan_image.height, y2)

    # Extract (possibly rectangular-shaped) patch
    patch = scan_image.data[y1:y2, x1:x2].copy()  # TODO: is copy() needed here?

    # Compute local coordinates (of the patch in the cell)
    local_x, local_y = 0, 0  # TODO: Implement this computation

    # Generate (square-shaped) padded output
    output = np.empty(shape=size, dtype=np.float64)
    output.fill(fill_value)
    output[local_y : local_y + patch.shape[0], local_x : local_x + patch.shape[1]] = (
        patch
    )

    return output
