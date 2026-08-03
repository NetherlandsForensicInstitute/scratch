from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from container_models.base import FloatArray2D, ImageRGB

DEFAULT_COLORMAP = "viridis"


def figure_to_array(fig: Figure) -> ImageRGB:
    """
    Convert a matplotlib figure to a numpy array.

    :param fig: Matplotlib figure to convert.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    canvas = cast(FigureCanvasAgg, fig.canvas)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()


def finish_figure(fig: Figure) -> ImageRGB:
    """Rasterize a figure to an RGB array and close it (always closes)."""
    try:
        return figure_to_array(fig)
    finally:
        plt.close(fig)


def render_single_panel(
    figsize: tuple[float, float],
    draw: Callable[[Figure, Axes], None],
    tight_layout_kwargs: dict | None = None,
) -> ImageRGB:
    """
    Render a single-axes figure to an RGB array.

    Creates the figure, hands (fig, ax) to ``draw``, applies tight_layout,
    then rasterizes and closes.

    :param figsize: (width, height) in inches.
    :param draw: Callback that plots onto the given figure and axes.
    :param tight_layout_kwargs: Extra kwargs for fig.tight_layout.
    :returns: RGB image as uint8 array.
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw(fig, ax)
    fig.tight_layout(**(tight_layout_kwargs or {}))
    return finish_figure(fig)


def finish_overview(
    fig: Figure,
    tight_layout_kwargs: dict,
    subplots_adjust_kwargs: dict,
) -> ImageRGB:
    """
    Apply tight_layout + subplots_adjust to an overview figure, then rasterize.

    :param fig: Figure to finalize.
    :param tight_layout_kwargs: kwargs for fig.tight_layout.
    :param subplots_adjust_kwargs: kwargs for fig.subplots_adjust.
    :returns: RGB image as uint8 array.
    """
    fig.tight_layout(**tight_layout_kwargs)
    fig.subplots_adjust(**subplots_adjust_kwargs)
    return finish_figure(fig)


def get_figure_dimensions(
    data_height: int,
    data_width: int,
    base_width: float = 10.0,
    height_padding: float = 1.0,
) -> tuple[float, float]:
    """
    Calculate figure dimensions based on data aspect ratio.

    :param data_height: Height of the data in pixels.
    :param data_width: Width of the data in pixels.
    :param base_width: Base figure width in inches.
    :param height_padding: Extra height in inches for title/labels/colorbar.
    :returns: Tuple of (figure_height, figure_width) in inches.
    """
    aspect_ratio = data_width / data_height
    fig_width = base_width
    fig_height = fig_width / aspect_ratio + height_padding
    return fig_height, fig_width


def side_by_side_gap_width(data_ref: FloatArray2D, data_comp: FloatArray2D) -> int:
    """Gap width in pixels between two surfaces plotted side by side."""
    return int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))


def get_height_ratios(metadata_height: float, *row_heights: float) -> list[float]:
    """
    Calculate normalized height ratios for a grid layout.

    :param metadata_height: Relative height for the metadata row.
    :param row_heights: Relative heights for the remaining rows.
    :returns: List of normalized height ratios for use with GridSpec.
    """
    heights = [metadata_height, *row_heights]
    total = sum(heights)
    return [h / total for h in heights]


def overview_figure_height(
    max_metadata_rows: int, base: float, minimum: float, maximum: float
) -> float:
    """Figure height for an overview: base + metadata rows, clamped to [min, max]."""
    return max(minimum, min(maximum, base + max_metadata_rows * 0.12))


def _fit_cell_label_fontsizes(
    ax: Axes,
    texts: list[Text],
    cell_w_um: float,
    fill: float = 0.85,
    min_fontsize: float = 3.0,
    max_fontsize: float = 11.0,
) -> None:
    """
    Shrink already-placed cell labels so each fits inside its cell.

    Measures the rendered width of each label and scales its font size to
    the cell width. A draw is forced first: with ``aspect="equal"`` the data
    box is only fitted inside the axes at draw time, so ``ax.transData`` is
    not final before then — reading it earlier overestimates the cell width
    (badly, for tall/narrow surfaces) and the labels come out too large.

    :param ax: Axes the labels were drawn on.
    :param texts: Text objects returned by ax.text.
    :param cell_w_um: Cell width in data units (µm).
    :param fill: Fraction of the cell width the text may occupy.
    :param min_fontsize: Lower clamp.
    :param max_fontsize: Upper clamp.
    """
    fig = ax.figure
    fig.canvas.draw()

    origin = ax.transData.transform((0.0, 0.0))
    edge = ax.transData.transform((cell_w_um, 0.0))
    cell_px = abs(edge[0] - origin[0])

    for text in texts:
        bbox = text.get_window_extent()
        if bbox.width <= 0:
            continue
        scaled = text.get_fontsize() * (cell_px * fill) / bbox.width
        text.set_fontsize(float(np.clip(scaled, min_fontsize, max_fontsize)))
