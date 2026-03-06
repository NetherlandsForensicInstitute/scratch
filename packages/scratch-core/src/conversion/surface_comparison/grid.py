import numpy as np
from scipy.signal import fftconvolve

from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import ComparisonParams
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
    # Extend forward
    c = origin_coord + cell_sz
    while c - cell_sz / 2 < image_sz:  # cell left edge is still within image
        centers.append(c)
        c += cell_sz
    # Extend backward
    c = origin_coord - cell_sz
    while c + cell_sz / 2 > 0:  # cell right edge is still within image
        centers.insert(0, c)
        c -= cell_sz
    return np.array(centers)


def _find_grid_origin(reference_map: ScanImage, params: ComparisonParams) -> np.ndarray:
    """
    Find the grid seed [x, y] that maximises total valid-data coverage.

    This is a faithful translation of ``cell_position_optim.m`` for the
    rectangle cell shape, implementing the default criteria chain:
    ``'sum cell score'`` followed by ``'min cell score'`` tie-breaker.

    Algorithm:
        1. Build a binary mask of valid (non-NaN) pixels.
        2. Pad the mask by (cell_size - 1) pixels on each side.
        3. Convolve with a cell-sized box kernel to get the per-pixel
           valid-pixel count for a cell centred at that pixel.
        4. Tile the convolved map into cell-sized blocks and stack them
           so that each position (i, j) in the tile represents one
           possible grid offset, and the stack dimension indexes
           individual cells at that offset.
        5. Discard cells below ``minimum_fill_fraction``.
        6. Apply scoring polynomial (default ``[1, 0]`` → score = fill × cell_area,
           i.e. total valid pixels).
        7. Primary criterion: maximise sum of cell scores across the stack.
        8. Tie-breaker: maximise the minimum cell score.
        9. Convert winning offset to physical coordinates.

    :param reference_map: The surface to divide into cells.
    :param params: Algorithm parameters (uses ``cell_size`` and
        ``minimum_fill_fraction``).
    :returns: Optimal origin coordinates [x, y] in meters, shape (2,).
    """
    height_map = reference_map.data
    pixel_spacing = reference_map.pixel_spacing  # [dx, dy] in meters

    # Cell size in pixels (integer)
    cell_size_px = meters_to_pixels(
        params.cell_size, pixel_spacing
    )  # [pixel_width, pixel_height]
    cell_area_px = int(np.prod(cell_size_px))

    # ---- Step 1: binary valid-pixel mask ----
    mask = (~np.isnan(height_map)).astype(np.float64)

    # ---- Step 2: pad by (cell_size - 1) on each side ----
    # MATLAB: map_crop(map, [-vOffset(1),-vOffset(1),-vOffset(2),-vOffset(2)], 0)
    # vOffset = vSizeP - 1
    pad_x = cell_size_px[0] - 1
    pad_y = cell_size_px[1] - 1
    mask_padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    # ---- Step 3: 2-D box-sum convolution ----
    # Equivalent to filter_nan(ones(cy,1), ones(1,cx), map, 0, 1) in MATLAB
    kernel = np.ones((cell_size_px[1], cell_size_px[0]), dtype=np.float64)
    count_map = fftconvolve(mask_padded, kernel, mode="valid")
    # After 'valid' convolution the output has the same size as the
    # original (unpadded) mask + 1 in each dimension because we padded
    # by (kernel - 1).  Trim to match the padded-map indexing used by MATLAB.
    # The MATLAB filter output has size = size(padded_map), and positions
    # correspond to the center of the kernel.
    # With 'valid' mode from scipy, the output is
    #   (padded_rows - ky + 1, padded_cols - kx + 1)
    # = (rows + 2*pad_y - cy + 1, cols + 2*pad_x - cx + 1)
    # = (rows + pad_y, cols + pad_x)
    # which is exactly what MATLAB's filter output contains (mapped into
    # the padded frame).

    # ---- Step 4: tile and stack ----
    # map_tile(map, vSizeP, [0,0], 0, NaN) tiles the count_map into
    # cell_size_px-sized blocks.  Each (i, j) in the tile grid represents
    # one possible grid offset, and the k-th layer is the k-th cell.
    rows_c, cols_c = count_map.shape
    n_tiles_y = rows_c // cell_size_px[1]
    n_tiles_x = cols_c // cell_size_px[0]

    if n_tiles_y == 0 or n_tiles_x == 0:
        # Surface too small for even one cell
        return np.zeros(2) * pixel_spacing

    # Trim to exact multiple of cell size
    trimmed = count_map[: n_tiles_y * cell_size_px[1], : n_tiles_x * cell_size_px[0]]

    # Reshape into (cell_py, n_tiles_y, cell_px, n_tiles_x) then
    # rearrange to (cell_py, cell_px, n_tiles_y * n_tiles_x)
    tiled = (
        trimmed.reshape(n_tiles_y, cell_size_px[1], n_tiles_x, cell_size_px[0])
        .transpose(1, 3, 0, 2)  # (pixel_height, pixel_width, n_tiles_y, n_tiles_x)
        .reshape(cell_size_px[1], cell_size_px[0], -1)
    )
    # tiled[oy, ox, k] = count of valid pixels for cell k when the grid
    # origin offset is (ox, oy) pixels.

    # ---- Step 5: normalise to fill fraction and discard low-fill cells ----
    fill = tiled / cell_area_px
    fill[fill < params.minimum_fill_fraction] = np.nan

    # Clip numerical artefacts
    fill = np.clip(fill, 0.0, 1.0)

    # ---- Step 6: scoring polynomial ----
    # Default vScorePol = [1, 0] → score = fill * 1 + 0 = fill
    # (which, multiplied by cell_area_px, equals the valid-pixel count)
    scores = np.copy(fill)  # polyval([1,0], fill) = fill

    # Guard: if no grid offset produces even a single cell above minimum_fill_fraction,
    # all scores are NaN. Fall back to offset (0, 0) — matching MATLAB's behaviour
    # in cell_position_optim when no candidate passes the fill threshold.
    if np.all(np.isnan(scores)):
        return np.zeros(2, dtype=np.float64)

    # ---- Step 7: primary criterion — maximise sum of cell scores ----
    sum_scores = np.nansum(scores, axis=2)  # (pixel_height, pixel_width)
    best_sum = np.nanmax(sum_scores)
    candidates = sum_scores >= best_sum

    # ---- Step 8: tie-breaker — maximise min cell score ----
    # Replace NaN with -inf before taking min so that positions with no valid
    # cells get -inf rather than triggering an all-NaN-slice warning.
    scores_for_min = np.where(np.isnan(scores), -np.inf, scores)
    min_scores = scores_for_min.min(axis=2)  # (pixel_height, pixel_width)
    min_scores[~candidates] = -np.inf
    best_min = np.nanmax(min_scores)
    final_mask = candidates & (min_scores >= best_min)

    # Pick first valid position (MATLAB: vi(1))
    oy, ox = np.argwhere(final_mask)[0]

    # ---- Step 9: convert pixel offset to physical coordinates ----
    # (oy, ox) is the tiling offset: tiled[oy, ox, k] addresses the k-th cell's
    # valid-pixel count when the grid starts at this offset.
    #
    # The first tile (k=0) has its centre at original-image pixel:
    #   centre_row = oy - (cell_size_px[1] / 2 - 0.5)
    #   centre_col = ox - (cell_size_px[0] / 2 - 0.5)
    #
    # This first-tile centre is used by generate_grid_centers as the origin from
    # which the full grid is extended in both directions.
    first_center_px = np.array(
        [
            ox - (cell_size_px[0] / 2 - 0.5),
            oy - (cell_size_px[1] / 2 - 0.5),
        ]
    )

    # Convert to physical coordinates
    origin_physical = first_center_px * pixel_spacing
    return origin_physical


def generate_grid_centers(
    reference_map: ScanImage, origin: np.ndarray, params: ComparisonParams
) -> np.ndarray:
    """
    Generate center coordinates for all cells in the grid.

    Uses the optimised ``origin`` (from :func:`_find_grid_origin`) as the first
    cell centre and extends the grid in each axis until cells would fall entirely
    outside the image.  This matches MATLAB's ``cell_position_optim`` + grid
    generation behaviour: the origin is the centre of the first tile found by
    the tiling optimisation, and subsequent cells are spaced by ``cell_size``.

    :param reference_map: surface map.
    :param origin: first cell centre [x, y] in meters from :func:`_find_grid_origin`.
    :param params: algorithm parameters.
    :returns: array of centre coordinates, shape (N, 2).
    """
    physical_size = reference_map.physical_size  # [width, height] in m
    cell_size = params.cell_size  # [cell_width, cell_height] in m

    x_coordinates = _axis_centers(origin[0], cell_size[0], physical_size[0])
    y_coordinates = _axis_centers(origin[1], cell_size[1], physical_size[1])

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
