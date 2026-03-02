"""
Cell grid origin optimisation.

Faithful translation of MATLAB ``cell_position_optim.m`` for rectangle cells.
Finds the pixel offset that maximises total valid-data coverage when the
reference surface is divided into a regular cell grid.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from conversion.surface_comparison.models import (
    SurfaceMap,
    ComparisonParams,
)


def find_optimal_cell_origin(
    reference_map: SurfaceMap, params: ComparisonParams
) -> np.ndarray:
    """
    Find the grid seed ``[x, y]`` that maximises total valid-data coverage.

    This is a faithful translation of ``cell_position_optim.m`` for the
    rectangle cell shape, implementing the default criteria chain:
    ``'sum cell score'`` followed by ``'min cell score'`` tie-breaker.

    Algorithm
    ---------
    1. Build a binary mask of valid (non-NaN) pixels.
    2. Pad the mask by ``(cell_size - 1)`` pixels on each side.
    3. Convolve with a cell-sized box kernel to get the per-pixel
       valid-pixel count for a cell centred at that pixel.
    4. Tile the convolved map into cell-sized blocks and stack them
       so that each position ``(i, j)`` in the tile represents one
       possible grid offset, and the stack dimension indexes
       individual cells at that offset.
    5. Discard cells below ``minimum_fill_fraction``.
    6. Apply scoring polynomial (default ``[1, 0]`` → score = fill).
    7. Primary criterion: maximise sum of cell scores across the stack.
    8. Tie-breaker: maximise the minimum cell score.
    9. Convert winning offset to physical coordinates.

    :param reference_map: The surface to divide into cells.
    :param params: Algorithm parameters (uses ``cell_size`` and
        ``minimum_fill_fraction``).
    :returns: Optimal origin coordinates ``[x, y]`` in metres, shape ``(2,)``.
    """
    height_map = reference_map.data
    pixel_spacing = reference_map.pixel_spacing  # [dx, dy]

    # Cell size in pixels [cx, cy] (x = col direction, y = row direction)
    cell_px = np.round(params.cell_size / pixel_spacing).astype(int)
    cell_area_px = int(np.prod(cell_px))

    # ---- Step 1: binary valid-pixel mask ----
    mask = (~np.isnan(height_map)).astype(np.float64)

    # ---- Step 2: pad by (cell_size - 1) on each side ----
    # MATLAB: vOffset = vSizeP - 1
    pad_x = cell_px[0] - 1
    pad_y = cell_px[1] - 1
    mask_padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    # ---- Step 3: 2-D box-sum convolution ----
    kernel = np.ones((cell_px[1], cell_px[0]), dtype=np.float64)
    count_map = fftconvolve(mask_padded, kernel, mode="valid")

    # ---- Step 4: tile and stack ----
    rows_c, cols_c = count_map.shape
    n_tiles_y = rows_c // cell_px[1]
    n_tiles_x = cols_c // cell_px[0]

    if n_tiles_y == 0 or n_tiles_x == 0:
        return np.zeros(2, dtype=np.float64)

    # Trim to exact multiple of cell size
    trimmed = count_map[: n_tiles_y * cell_px[1], : n_tiles_x * cell_px[0]]

    # Reshape: each (oy, ox) position in the tile = one possible grid offset,
    # stack dimension = individual cells at that offset
    tiled = (
        trimmed.reshape(n_tiles_y, cell_px[1], n_tiles_x, cell_px[0])
        .transpose(1, 3, 0, 2)
        .reshape(cell_px[1], cell_px[0], -1)
    )

    # ---- Step 5: normalise to fill fraction and discard low-fill cells ----
    fill = tiled / cell_area_px
    fill[fill < params.minimum_fill_fraction] = np.nan
    fill = np.clip(fill, 0.0, 1.0)

    # ---- Step 6: scoring polynomial [1, 0] → score = fill ----
    scores = np.copy(fill)

    # ---- Step 7: primary criterion — maximise sum of cell scores ----
    sum_scores = np.nansum(scores, axis=2)
    best_val = np.nanmax(sum_scores)
    candidates = sum_scores >= best_val

    # ---- Step 8: tie-breaker — maximise min cell score ----
    min_scores = np.nanmin(scores, axis=2)
    min_scores[~candidates] = -np.inf
    best_min = np.nanmax(min_scores)
    final_mask = candidates & (min_scores >= best_min)

    # Pick first valid position (MATLAB: vi(1))
    oy, ox = np.argwhere(final_mask)[0]

    # ---- Step 9: convert pixel offset to physical coordinates ----
    # The offset (ox, oy) is 0-based in the padded count_map frame.
    # The origin of the grid in physical coordinates is:
    #   origin = offset * pixel_spacing
    # This gives the position of the first cell's corner (top-left).
    origin = np.array([ox * pixel_spacing[0], oy * pixel_spacing[1]])

    return origin
