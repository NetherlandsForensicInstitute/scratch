import numpy as np
from conversion.surface_comparison.models import SurfaceMap, ComparisonParams


def find_optimal_cell_origin(
    reference_map: SurfaceMap, params: ComparisonParams
) -> np.ndarray:
    """
    Find the grid seed [x, y] that maximizes total valid data coverage.

    :param reference_map: the surface to divide into cells.
    :param params: algorithm parameters.
    :returns: optimal origin coordinates [x, y] in micrometers, shape (2,).
    """
    mask = ~np.isnan(reference_map.height_map)
    rows, columns = reference_map.height_map.shape
    pixel_width, pixel_height = reference_map.pixel_spacing

    cell_width_pixels = int(params.cell_size[0] / pixel_width)
    cell_height_pixels = int(params.cell_size[1] / pixel_height)

    best_count = -1
    best_origin_pixels = np.zeros(2, dtype=int)

    for offset_y in range(0, cell_height_pixels, max(1, cell_height_pixels // 4)):
        for offset_x in range(0, cell_width_pixels, max(1, cell_width_pixels // 4)):
            current_count = np.sum(
                mask[
                    offset_y:rows:cell_height_pixels, offset_x:columns:cell_width_pixels
                ]
            )
            if current_count > best_count:
                best_count = current_count
                best_origin_pixels = np.array([offset_x, offset_y])

    return best_origin_pixels * reference_map.pixel_spacing


def generate_cell_centers(
    reference_map: SurfaceMap, origin: np.ndarray, params: ComparisonParams
) -> np.ndarray:
    """
    Generate center coordinates for all cells in the grid.

    :param reference_map: surface map.
    :param origin: optimal starting point [x, y] in micrometers, shape (2,).
    :param params: algorithm parameters.
    :returns: array of center coordinates, shape (N, 2).
    """
    physical_width, physical_height = reference_map.physical_size
    cell_width, cell_height = params.cell_size

    x_coordinates = np.arange(origin[0] + cell_width / 2, physical_width, cell_width)
    y_coordinates = np.arange(origin[1] + cell_height / 2, physical_height, cell_height)

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
