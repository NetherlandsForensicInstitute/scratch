import numpy as np
from scipy.signal import fftconvolve

from conversion.surface_comparison.models import SurfaceMap, ComparisonParams
from conversion.surface_comparison.utils import meters_to_pixels


def _axis_centers(origin_coord: float, cell_sz: float, image_sz: float) -> np.ndarray:
    """
    Extend a 1-D grid from an origin in both directions while cells overlap the image.

    :param origin_coord: Center of the first cell in meters.
    :param cell_sz: Cell size in meters.
    :param image_sz: Image extent in meters.
    :returns: Sorted array of cell center coordinates in meters.
    """
    centers = [origin_coord]
    c = origin_coord + cell_sz
    while c - cell_sz / 2 < image_sz:
        centers.append(c)
        c += cell_sz
    c = origin_coord - cell_sz
    while c + cell_sz / 2 > 0:
        centers.insert(0, c)
        c -= cell_sz
    return np.array(centers)


def _find_grid_origin(
    reference_map: SurfaceMap, params: ComparisonParams
) -> np.ndarray:
    """
    Find the grid seed [x, y] that maximises total valid-data coverage.

    Faithful translation of ``cell_position_optim.m``: 'sum cell score' primary
    criterion, 'min cell score' tie-breaker.

    :param reference_map: The surface to divide into cells.
    :param params: Algorithm parameters (uses ``cell_size`` and ``minimum_fill_fraction``).
    :returns: Optimal origin coordinates [x, y] in meters, shape (2,).
    """
    height_map = reference_map.data
    pixel_spacing = reference_map.pixel_spacing

    cell_size_px = meters_to_pixels(params.cell_size, pixel_spacing)
    cell_area_px = int(np.prod(cell_size_px))

    mask = (~np.isnan(height_map)).astype(np.float64)

    pad_x = cell_size_px[0] - 1
    pad_y = cell_size_px[1] - 1
    mask_padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    kernel = np.ones((cell_size_px[1], cell_size_px[0]), dtype=np.float64)
    count_map = fftconvolve(mask_padded, kernel, mode="valid")

    rows_c, cols_c = count_map.shape
    n_tiles_y = rows_c // cell_size_px[1]
    n_tiles_x = cols_c // cell_size_px[0]

    if n_tiles_y == 0 or n_tiles_x == 0:
        return np.zeros(2) * pixel_spacing

    trimmed = count_map[: n_tiles_y * cell_size_px[1], : n_tiles_x * cell_size_px[0]]

    tiled = (
        trimmed.reshape(n_tiles_y, cell_size_px[1], n_tiles_x, cell_size_px[0])
        .transpose(1, 3, 0, 2)
        .reshape(cell_size_px[1], cell_size_px[0], -1)
    )

    fill = tiled / cell_area_px
    fill[fill < params.minimum_fill_fraction] = np.nan
    fill = np.clip(fill, 0.0, 1.0)

    scores = np.copy(fill)

    if np.all(np.isnan(scores)):
        return np.zeros(2, dtype=np.float64)

    sum_scores = np.nansum(scores, axis=2)
    best_sum = np.nanmax(sum_scores)
    candidates = sum_scores >= best_sum

    scores_for_min = np.where(np.isnan(scores), -np.inf, scores)
    min_scores = scores_for_min.min(axis=2)
    min_scores[~candidates] = -np.inf
    best_min = np.nanmax(min_scores)
    final_mask = candidates & (min_scores >= best_min)

    oy, ox = np.argwhere(final_mask)[0]

    first_center_px = np.array(
        [
            ox - (cell_size_px[0] / 2 - 0.5),
            oy - (cell_size_px[1] / 2 - 0.5),
        ]
    )
    return first_center_px * pixel_spacing


def generate_grid_centers(
    reference_map: SurfaceMap, origin: np.ndarray, params: ComparisonParams
) -> np.ndarray:
    """
    Generate center coordinates for all cells in the grid.

    :param reference_map: surface map.
    :param origin: first cell centre [x, y] in meters from :func:`_find_grid_origin`.
    :param params: algorithm parameters.
    :returns: array of centre coordinates, shape (N, 2).
    """
    physical_size = reference_map.physical_size
    cell_size = params.cell_size

    x_coordinates = _axis_centers(origin[0], cell_size[0], physical_size[0])
    y_coordinates = _axis_centers(origin[1], cell_size[1], physical_size[1])

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
