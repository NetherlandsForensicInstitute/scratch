from typing import Sequence, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import mega

from container_models.base import FloatArray2D
from conversion.plots.utils import DEFAULT_COLORMAP, _fit_fontsize
from conversion.plots.on_axes import _robust_color_limits

from conversion.surface_comparison.models import Cell


def _draw_cell_labels(
    ax: Axes,
    cells: Sequence[Cell],
    cell_label_prefix: str,
    show_all_cells: bool,
    space: Literal["reference", "comparison"] = "comparison",
) -> None:
    """
    Draw labeled cell rectangles on axes, colored by CMC status.

    CMC cells (above threshold) are drawn in black, non-CMC cells in red.
    CMC cells are drawn first so red outlines are not hidden by adjacent borders.
    In "comparison" space, labels are rotated to match the cell's angle_deg
    so the text orientation follows the (possibly rotated) cell.

    :param ax: Matplotlib axes to draw on.
    :param cells: cells to draw
    :param cell_label_prefix: Label prefix for cells ("A" or "B").
    :param show_all_cells: If True, show all cells. If False, only show CMC cells.
    :param space: ``"reference"`` draws cells at their grid positions
        (``center_reference``, no rotation); ``"comparison"`` draws them at
        their matched positions (``center_comparison`` / ``angle_deg``,
        counter-clockwise positive, standard math/plot convention with
        y-axis increasing upward).
    """
    cmc_cells: list[tuple[int, Cell]] = []
    non_cmc_cells: list[tuple[int, Cell]] = []

    for idx, cell in enumerate(cells, start=1):
        if cell.is_congruent:
            cmc_cells.append((idx, cell))
        elif show_all_cells:
            non_cmc_cells.append((idx, cell))

    w_um, h_um = cells[0].cell_size_um
    half_w_um, half_h_um = w_um / 2, h_um / 2
    base_corners = np.array(
        [[-half_w_um, -half_h_um], [half_w_um, -half_h_um], [half_w_um, half_h_um], [-half_w_um, half_h_um]]
    )

    for color, labeled_cells in [("black", cmc_cells), ("red", non_cmc_cells)]:
        for idx, cell in labeled_cells:
            if space == "reference":
                # Regular grid position, axis-aligned (no rotation).
                cx = cell.center_reference[0] * 1e6
                cy = cell.center_reference[1] * 1e6
                corners_rotated = base_corners.copy()
                text_rotation = 0.0
            else:
                # Matched position with the per-cell rotation applied.
                cx = cell.center_comparison[0] * 1e6
                cy = cell.center_comparison[1] * 1e6
                angle = np.deg2rad(-cell.angle_deg)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                # Counter-clockwise rotation matrix (standard math convention,
                # positive angle_deg = CCW when y-axis increases upward).
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                corners_rotated = base_corners @ rot.T
                text_rotation = -cell.angle_deg

            corners_translated = corners_rotated + [cx, cy]

            xs = np.append(corners_translated[:, 0], corners_translated[0, 0])
            ys = np.append(corners_translated[:, 1], corners_translated[0, 1])
            ax.plot(xs, ys, color=color, linestyle="-", linewidth=1.0)

            label = f"{cell_label_prefix}{idx}"
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=_fit_fontsize(ax, label, w_um),
                color=color,
                fontweight="bold",
                rotation=text_rotation,
                rotation_mode="anchor",
            )


def plot_cell_overlay_on_axes(
    ax: Axes,
    data: FloatArray2D,
    scale: float,
    cells: Sequence[Cell],
    cell_label_prefix: str = "A",
    show_all_cells: bool = True,
    space: Literal["reference", "comparison"] = "comparison",
    color_sigma: float = 2.5,
) -> AxesImage:
    """
    Plot surface with cell grid overlay on given axes.

    Follows the MATLAB plot_cells convention: cells above the similarity
    threshold (CMC cells) are drawn with black outlines and labels, while
    cells below the threshold are drawn with red outlines and labels.

    Color scaling clips to median ± color_sigma * (1.4826*MAD) of the data,
    so a few extreme outlier pixels don't blow out the contrast for the rest
    of the surface.

    :param ax: Matplotlib axes to plot on.
    :param data: Surface data in meters.
    :param scale: Pixel scale in meters.
    :param cells: cells to plot
    :param cell_label_prefix: Label prefix for cells ("A" for reference, "B" for compared).
    :param show_all_cells: If True, show all cells. If False, only show CMC cells.
    :param space: Which positions to draw the cells at. ``"reference"`` draws
        each cell at its regular-grid position (``center_reference``) with no
        rotation — use for the reference surface. ``"comparison"`` draws each
        cell at its matched, rotated position (``center_comparison`` /
        ``angle_deg``) — use for the moved compared surface.
    :param color_sigma: Number of robust standard deviations (median ±
        color_sigma * 1.4826*MAD) to clip the color scale at. Defaults to 3.0.
    """
    height, width = data.shape

    extent = (0, width * scale * mega, 0, height * scale * mega)

    data_um = data * mega
    vmin, vmax = _robust_color_limits(data_um, k=color_sigma)

    im = ax.imshow(
        data_um,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    _draw_cell_labels(
        ax,
        cells=cells,
        cell_label_prefix=cell_label_prefix,
        show_all_cells=show_all_cells,
        space=space,
    )

    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.tick_params(labelsize=10)

    return im


def _plot_cell_heatmap_on_axes(
    ax: Axes,
    fig: Figure,
    cell_correlations: FloatArray2D,
    cells: Sequence[Cell],
    surface_extent_um: tuple[float, float],
    cell_label_prefix: str = "A",
) -> None:
    """
    Plot cell correlation heatmap on given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param cell_correlations: Grid of per-cell correlation values (n_rows, n_cols).
    :param cells: Cell results from the CMC pipeline.
    :param surface_extent_um: (width, height) of the surface in µm.
    :param cell_label_prefix: Prefix for cell labels.
    """
    w_um, h_um = surface_extent_um

    im = ax.imshow(
        cell_correlations,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=(0, w_um, 0, h_um),
        vmin=0,
        vmax=1,
    )
    tile_w_um = w_um / cell_correlations.shape[1]

    for idx, cell in enumerate(cells, start=1):
        if np.isnan(cell.best_score):
            continue

        cx = cell.center_reference[0] * 1e6
        cy = cell.center_reference[1] * 1e6
        color = "blue" if cell.is_congruent else "red"
        label = f"{cell_label_prefix}{idx}"
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=_fit_fontsize(ax, label, tile_w_um),
            color=color,
            fontweight="bold",
        )

    ax.set_title("Cell ACCF Distribution", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
