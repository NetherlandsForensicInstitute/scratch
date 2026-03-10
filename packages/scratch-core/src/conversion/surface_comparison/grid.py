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


def extract_patch(
    scan_image: ScanImage,
    coordinates: tuple[int, int],  # Top-left coordinates
    size: tuple[int, int],  # Usually a square
    fill_value: float = np.nan,
) -> FloatArray2D:
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


def generate_grid(scan_image: ScanImage, params: ComparisonParams) -> list[GridCell]:
    cell_width = _convert_meters_to_pixels(params.cell_size[0], scan_image.scale_x)
    cell_height = _convert_meters_to_pixels(params.cell_size[1], scan_image.scale_y)

    # TODO: Implement some complicated grid generating procedure. For now let's generate something simple
    output = []
    y, x = 0, 0
    while y < scan_image.height:
        while x < scan_image.width:
            cell_data = extract_patch(
                scan_image=scan_image,
                coordinates=(x, y),
                size=(cell_width, cell_height),
                fill_value=np.nan,
            )
            center = int(x + cell_width // 2), int(y + cell_height // 2)
            cell = GridCell(
                center=center,
                size=(cell_width, cell_height),
                cell_data=cell_data,
                grid_search_params=GridSearchParams(),
            )
            if cell.fill_fraction < params.minimum_fill_fraction:
                # Check cell validity with respect to the imposed domain constraints
                continue
            output.append(cell)
            x += cell_width

        y += cell_height

    return output
