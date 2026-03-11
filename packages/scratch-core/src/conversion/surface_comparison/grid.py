import math

import numpy as np

from container_models.base import FloatArray2D
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import GridCell, GridSearchParams
from conversion.surface_comparison.utils import convert_meters_to_pixels


def generate_grid(
    scan_image: ScanImage, cell_size: tuple[float, float], minimum_fill_fraction: float
) -> list[GridCell]:
    """
    Generate a centered grid of cells covering the image.

    The grid is symmetrically centered on the image using the MATLAB even/odd
    seed logic. Cells with insufficient valid data are filtered out.

    :param scan_image: reference scan image
    :param cell_size: size of the cells of the grid in meters
    :param minimum_fill_fraction: minimum fraction of valid data of each cell
    :return: list of valid grid cells
    """
    cell_width, cell_height = convert_meters_to_pixels(cell_size, scan_image.scale_x)
    xs = _tile_axis(scan_image.width, cell_width)
    ys = _tile_axis(scan_image.height, cell_height)

    output = []
    for y in ys:
        for x in xs:
            cell_data = extract_patch(
                scan_image=scan_image,
                coordinates=(x, y),
                patch_size=(cell_width, cell_height),
                fill_value=np.nan,
            )
            cell = GridCell(top_left=(x, y), cell_data=cell_data, grid_search_params=GridSearchParams())
            if cell.fill_fraction < minimum_fill_fraction:
                continue
            output.append(cell)
    return output


def extract_patch(
    scan_image: ScanImage,
    coordinates: tuple[int, int],
    patch_size: tuple[int, int],
    fill_value: float = np.nan,
) -> FloatArray2D:
    """Extract a rectangular patch from a scan image, padding with fill_value
    where the patch extends beyond the image boundaries.

    :param scan_image: source image to extract from
    :param coordinates: (x, y) top-left corner of the patch, may be negative
    :param patch_size: (width, height) of the output patch
    :param fill_value: value to use for out-of-bounds pixels
    :return: 2D array of shape (height, width)
    """
    x, y = coordinates
    width, height = patch_size

    # Find the overlap between the patch and the image
    row_start = max(y, 0)
    row_end = min(y + height, scan_image.height)
    col_start = max(x, 0)
    col_end = min(x + width, scan_image.width)

    if row_start >= row_end or col_start >= col_end:
        raise ValueError(
            f"Patch at ({x}, {y}) with size ({width}, {height}) "
            f"has no overlap with image of size ({scan_image.width}, {scan_image.height})"
        )

    # Get the patch
    patch = scan_image.data[row_start:row_end, col_start:col_end]

    # Pad where needed
    pad_top = row_start - y
    pad_bottom = y + height - row_end
    pad_left = col_start - x
    pad_right = x + width - col_end

    padded = np.pad(
        patch,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=fill_value,
    )
    return padded


def _tile_axis(image_size: int, cell_size: int) -> list[int]:
    """Generate top-left coordinates for cells along one axis.

    Places cells symmetrically around the midpoint of the image. When an
    odd number of cells fits, one cell is centered on the midpoint. When
    even, two cells straddle it.

    :param image_size: image size in pixels along the axis
    :param cell_size: cell size in pixels along the axis
    :return: sorted list of top-left coordinates
    """
    n = math.ceil(image_size / cell_size)
    if n % 2 == 1:
        anchor = image_size / 2
    else:
        anchor = image_size / 2 - cell_size / 2

    offsets = np.arange(n) - (n - 1) // 2
    top_lefts = np.round(anchor - cell_size / 2 + offsets * cell_size).astype(int)

    return top_lefts.tolist()
