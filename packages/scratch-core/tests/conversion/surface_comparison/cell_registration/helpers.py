"""Shared test helpers for the cell-registration test suite.

All helpers are pure functions and carry no pytest imports so this module is
safe to import from any test file without side-effects.
"""

from __future__ import annotations

import numpy as np

from container_models.base import DepthData, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.resample import resize_array_2d_nan_aware
from conversion.surface_comparison.models import (
    Cell,
    ComparisonParams,
    GridCell,
    GridSearchParams,
)

from .plot_utils import (
    plot_rotated_squares,
    plot_side_by_side,
)


def make_surface(
    height: int,
    width: int,
    scale: float = 1.0,
    nan_ratio: float = 0.0,
    seed: int = 0,
) -> DepthData:
    """
    Return a deterministic, non-periodic 2-D height map with structure at several scales.

    A smooth global trend plus band-limited random layers at octave scales, plus fine noise.
    The multi-scale structure matters: a surface of smooth trend plus white noise localizes a
    cell fine at full resolution, but the noise averages away under downsampling and leaves
    nothing for a coarse search to lock onto. A deterministic ripple survives downsampling but
    repeats, which is worse still - it produces many near-equal matches. Random layers give
    features that both survive averaging and stay unique.

    :param height: Number of rows.
    :param width: Number of columns.
    :param scale: Multiplicative scale applied to the whole array - use e.g. ``1e-6`` to
        simulate µm-scale surface data.
    :param nan_ratio: The ratio of NaN values randomly generated.
    :param seed: Random seed.
    :returns: ``(height, width)`` float64 array.
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, height)
    x = np.linspace(0.0, 1.0, width)
    Y, X = np.meshgrid(y, x, indexing="ij")

    surface = np.exp(-3.0 * Y) * np.cos(7.391 * X) + np.exp(-2.0 * X) * np.sin(
        5.123 * Y
    )

    for octave, amplitude in ((8, 0.6), (16, 0.3), (32, 0.15)):
        control = rng.standard_normal((octave, octave))
        rows = np.linspace(0, octave - 1, height)
        cols = np.linspace(0, octave - 1, width)
        row_index = np.clip(rows.astype(int), 0, octave - 2)
        col_index = np.clip(cols.astype(int), 0, octave - 2)
        row_frac = (rows - row_index)[:, None]
        col_frac = (cols - col_index)[None, :]
        surface = surface + amplitude * (
            control[row_index][:, col_index] * (1 - row_frac) * (1 - col_frac)
            + control[row_index + 1][:, col_index] * row_frac * (1 - col_frac)
            + control[row_index][:, col_index + 1] * (1 - row_frac) * col_frac
            + control[row_index + 1][:, col_index + 1] * row_frac * col_frac
        )

    surface = surface + rng.standard_normal((height, width)) * 0.05
    if 0.0 < nan_ratio < 1:
        surface[rng.uniform(size=surface.shape) < nan_ratio] = np.nan
    return (surface * scale).astype(np.float64)


def make_scan_image(
    height: int,
    width: int,
    pixel_size: float = 1e-6,
    scale: float = 1.0,
    nan_ratio: float = 0.0,
    seed: int = 0,
) -> ScanImage:
    """Construct a :class:`ScanImage` wrapping :func:`make_surface` output."""
    data = make_surface(
        height=height, width=width, scale=scale, nan_ratio=nan_ratio, seed=seed
    )
    return ScanImage(data=data, scale_x=pixel_size, scale_y=pixel_size)


def make_grid_cell(
    data: FloatArray2D,
    top_left: tuple[int, int] = (0, 0),
    nan_fill_value: float | None = None,
) -> GridCell:
    """Wrap a 2-D array in a :class:`GridCell` with a fresh :class:`GridSearchParams`."""
    return GridCell(
        top_left=top_left,
        cell_data=data.copy(),
        grid_search_params=GridSearchParams(),
        nan_fill_value=nan_fill_value,
    )


def downsample(image: FloatArray2D, factor: float) -> FloatArray2D:
    """NaN-aware area-average shrink, matching what the coarse stage does to both images."""
    height, width = image.shape
    new_shape = (int(np.ceil(height / factor)), int(np.ceil(width / factor)))
    return resize_array_2d_nan_aware(image, new_shape, interpolation="area")


def identity_params() -> ComparisonParams:
    """Return :class:`ComparisonParams` configured for a zero-angle identity test."""
    return ComparisonParams(
        minimum_fill_fraction=0.5,
        correlation_threshold=0.5,
        search_angle_min=-60.0,
        search_angle_max=60.0,
        search_angle_step=60.0,
    )


def plot_cell_registration_results(
    reference_image: ScanImage, comparison_image: ScanImage, cells: list[Cell]
):
    ref_plot = plot_rotated_squares(
        image=reference_image.data,
        cells=cells,
        pixel_size=reference_image.scale_x,
        mode="reference",
    )
    comp_plot = plot_rotated_squares(
        image=comparison_image.data,
        cells=cells,
        pixel_size=comparison_image.scale_x,
        mode="comparison",
    )
    plot_side_by_side(
        img1=ref_plot, title1="Reference", img2=comp_plot, title2="Comparison"
    )
