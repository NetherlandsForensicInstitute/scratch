import numpy as np
from scipy.signal import fftconvolve

from conversion.surface_comparison.models import SurfaceMap, ComparisonParams


def find_grid_origin(reference_map: SurfaceMap, params: ComparisonParams) -> np.ndarray:
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
    :returns: Optimal origin coordinates [x, y] in micrometers, shape (2,).
    """
    height_map = reference_map.height_map
    pixel_spacing = reference_map.pixel_spacing  # [dx, dy]

    # Cell size in pixels (integer)
    cell_px = np.round(params.cell_size / pixel_spacing).astype(int)  # [cx, cy]
    cell_area_px = int(np.prod(cell_px))

    # ---- Step 1: binary valid-pixel mask ----
    mask = (~np.isnan(height_map)).astype(np.float64)

    # ---- Step 2: pad by (cell_size - 1) on each side ----
    # MATLAB: map_crop(map, [-vOffset(1),-vOffset(1),-vOffset(2),-vOffset(2)], 0)
    # vOffset = vSizeP - 1
    pad_x = cell_px[0] - 1
    pad_y = cell_px[1] - 1
    mask_padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    # ---- Step 3: 2-D box-sum convolution ----
    # Equivalent to filter_nan(ones(cy,1), ones(1,cx), map, 0, 1) in MATLAB
    kernel = np.ones((cell_px[1], cell_px[0]), dtype=np.float64)
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
    # cell_px-sized blocks.  Each (i, j) in the tile grid represents
    # one possible grid offset, and the k-th layer is the k-th cell.
    rows_c, cols_c = count_map.shape
    n_tiles_y = rows_c // cell_px[1]
    n_tiles_x = cols_c // cell_px[0]

    if n_tiles_y == 0 or n_tiles_x == 0:
        # Surface too small for even one cell
        return np.zeros(2) * pixel_spacing

    # Trim to exact multiple of cell size
    trimmed = count_map[: n_tiles_y * cell_px[1], : n_tiles_x * cell_px[0]]

    # Reshape into (cell_py, n_tiles_y, cell_px, n_tiles_x) then
    # rearrange to (cell_py, cell_px, n_tiles_y * n_tiles_x)
    tiled = (
        trimmed.reshape(n_tiles_y, cell_px[1], n_tiles_x, cell_px[0])
        .transpose(1, 3, 0, 2)  # (cell_py, cell_px, n_tiles_y, n_tiles_x)
        .reshape(cell_px[1], cell_px[0], -1)
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
    sum_scores = np.nansum(scores, axis=2)  # (cell_py, cell_px)
    best_val = np.nanmax(sum_scores)
    candidates = sum_scores >= best_val  # tolerance = 0 by default

    # ---- Step 8: tie-breaker — maximise min cell score ----
    min_scores = np.nanmin(scores, axis=2)  # (cell_py, cell_px)
    min_scores[~candidates] = -np.inf
    best_min = np.nanmax(min_scores)
    final_mask = candidates & (min_scores >= best_min)

    # Pick first valid position (MATLAB: vi(1))
    oy, ox = np.argwhere(final_mask)[0]

    # ---- Step 9: convert pixel offset to physical coordinates ----
    # MATLAB: vCellPosition = trans_ind2sub(..., vi(1)) then
    #         pix2pos(vCellPosition - vOffset, vCenterG, {}, vCenterL, vPixSep)
    #
    # The offset in the padded frame is (ox, oy).  Subtracting vOffset
    # brings us back to the original map frame (0-indexed).  The even-size
    # correction adds 0.5 for even cell dimensions.
    pos_px = np.array([ox, oy], dtype=float)

    # Even-size correction (MATLAB adds 0.5 for even cell dimensions)
    for i in range(2):
        if cell_px[i] % 2 == 0:
            pos_px[i] += 0.5

    # Convert to physical coordinates
    # In MATLAB: pix2pos(pixel - offset, CenterG, {}, CenterL, PixSep)
    # For the default case where CenterG and CenterL are the map center:
    #   pos = CenterG + (pixel - CenterL) * PixSep
    # But since cell_position_optim works with zero-based offsets relative
    # to the map origin (pixel [0,0] = top-left of map), and the final
    # conversion uses the reference map's coordinate system, we compute:
    origin_physical = (
        reference_map.global_center
        + (pos_px - reference_map.global_center / pixel_spacing) * pixel_spacing
    )
    # Simplified: this is just pos_px * pixel_spacing when the global center
    # aligns with the local center (which is the typical case for NFI data
    # where vCenterG = vCenterL for the reference).
    #
    # For maximum fidelity, we use:
    #   global_pos = vCenterG + (pix_pos - vCenterL/vPixSep) * vPixSep
    # But vCenterL is in physical units and represents the center pixel's
    # physical offset from the map origin, so:
    origin_physical = pos_px * pixel_spacing

    return origin_physical


def generate_grid_centers(
    reference_map: SurfaceMap, origin: np.ndarray, params: ComparisonParams
) -> np.ndarray:
    """
    Generate center coordinates for all cells in the grid.

    When ``bCellGlobalCenter=1`` (the MATLAB default), the cell grid is centered
    on the map's global center rather than starting from the top-left corner.
    The number of cells per axis is ``ceil(physical_size / cell_size)``, which
    may place some cells partially outside the image boundary — those cells are
    handled by clipping during patch extraction.

    The ``origin`` offset (from :func:`find_grid_origin`) is not applied here
    because MATLAB's ``bCellGlobalCenter=1`` flag overrides the optimised origin
    and locks the grid to the image center.

    :param reference_map: surface map.
    :param origin: optimal starting point [x, y] in micrometers, shape (2,).
        Currently unused when using the centered-grid convention.
    :param params: algorithm parameters.
    :returns: array of center coordinates, shape (N, 2).
    """
    import math

    physical_size = reference_map.physical_size  # [width, height] in µm
    cell_size = params.cell_size  # [cell_w, cell_h] in µm
    center = reference_map.global_center  # [cx, cy] in µm

    # Number of cells per axis: enough to cover the full physical extent
    n_cells_x = math.ceil(physical_size[0] / cell_size[0])
    n_cells_y = math.ceil(physical_size[1] / cell_size[1])

    # Cell centers are symmetric around global_center:
    #   offset_i = (i - n/2 + 0.5) * cell_size  for i in 0..n-1
    x_coordinates = np.array(
        [center[0] + (i - n_cells_x / 2 + 0.5) * cell_size[0] for i in range(n_cells_x)]
    )
    y_coordinates = np.array(
        [center[1] + (i - n_cells_y / 2 + 0.5) * cell_size[1] for i in range(n_cells_y)]
    )

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)
