"""Shared test helpers for the cell-registration test suite.

All helpers are pure functions and carry no pytest imports so this module is
safe to import from any test file without side-effects.
"""

import numpy as np

from container_models.base import FloatArray2D, DepthData
from container_models.scan_image import ScanImage
from conversion.surface_comparison.models import (
    ComparisonParams,
    GridCell,
    GridSearchParams,
    Cell,
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
    Return a deterministic, non-periodic 2-D height map.

    Built from a sum of decaying exponentials with irrational frequencies so
    that no integer pixel shift produces an exact repeat.

    :param height: Number of rows.
    :param width: Number of columns.
    :param scale: Multiplicative scale applied to the whole array — use e.g.
        ``1e-6`` to simulate µm-scale surface data.
    :param nan_ratio: The ratio of NaN values randomly generated.
    :param seed: Random seed for the small noise component.
    :returns: ``(height, width)`` float64 array.
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, height)
    x = np.linspace(0.0, 1.0, width)
    Y, X = np.meshgrid(y, x, indexing="ij")

    surface = (
        np.exp(-3.0 * Y) * np.cos(7.391 * X)
        + np.exp(-2.0 * X) * np.sin(5.123 * Y)
        + rng.standard_normal((height, width)) * 0.05
    )
    if 0.0 < nan_ratio < 1:
        surface[np.random.uniform(size=surface.shape) < nan_ratio] = np.nan
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
    nan_fill_value: float = np.nan,
) -> GridCell:
    """Wrap a 2-D array in a :class:`GridCell` with a fresh :class:`GridSearchParams`."""
    return GridCell(
        top_left=top_left,
        cell_data=data.copy(),
        grid_search_params=GridSearchParams(),
        nan_fill_value=nan_fill_value,
    )


def identity_params(
    cell_size_px: int,
    pixel_size: float = 1e-5,
) -> ComparisonParams:
    """
    Return :class:`ComparisonParams` configured for a zero-angle identity test.

    :param cell_size_px: Cell size in pixels; converted to meters via *pixel_size*.
    :param pixel_size: Pixel size in metres per pixel.
    """
    cell_size_m = cell_size_px * pixel_size
    return ComparisonParams(
        cell_size=(cell_size_m, cell_size_m),
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
