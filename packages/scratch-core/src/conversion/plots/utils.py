import textwrap
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray


def figure_to_array(fig: Figure) -> np.ndarray:
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


def get_fig_dimensions(height: int, width: int) -> tuple[float, float]:
    """Calculate figure dimensions based on data aspect ratio."""
    aspect_ratio = width / height
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    return fig_height + 1.0, fig_width


def plot_profiles_on_axes(
    ax: Axes,
    profile_ref: NDArray,
    profile_comp: NDArray,
    scale: float,
    score: float,
    title: str,
) -> None:
    """
    Plot two aligned profiles on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param profile_ref: Reference profile (aligned, 1D).
    :param profile_comp: Compared profile (aligned, 1D).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient.
    :param title: Prefix for the title before the correlation value.
    """
    x1 = np.arange(len(profile_ref)) * scale * 1e6  # µm
    x2 = np.arange(len(profile_comp)) * scale * 1e6

    y1 = profile_ref * 1e6  # µm
    y2 = profile_comp * 1e6

    ax.plot(x1, y1, "b-", label="Reference Profile A", linewidth=1.5)
    ax.plot(x2, y2, "r-", label="Compared Profile B", linewidth=1.5)

    ax.set_xlabel("Profile Length [µm]", fontsize=11)
    ax.set_ylabel("Profile Height [µm]", fontsize=11)
    ax.set_title(f"{title}: {score:.5f}", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_depthmap_on_axes(
    ax: Axes,
    fig: Figure,
    data: NDArray,
    scale: float,
    title: str,
    shrink_colorbar: float = 1.0,
) -> None:
    """
    Plot a depth map on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data: Data to plot in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param shrink_colorbar: Shrink factor for colorbar (0-1).
    """
    height, width = data.shape
    extent = (0, width * scale * 1e6, 0, height * scale * 1e6)

    im = ax.imshow(
        data * 1e6,
        cmap=DEFAULT_COLORMAP,
        aspect="equal",
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    cbar = fig.colorbar(im, ax=ax, label="µm", shrink=shrink_colorbar)
    cbar.ax.tick_params(labelsize=10)


def _plot_side_by_side_on_axes(
    ax: Axes,
    fig: Figure,
    data_ref: NDArray,
    data_comp: NDArray,
    scale: float,
    title: str = "Reference Surface A / Moved Compared Surface B",
    shrink_colorbar: float = 1.0,
) -> None:
    """
    Plot two surfaces side by side on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data_ref: Reference data in meters.
    :param data_comp: Compared data in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param shrink_colorbar: Shrink factor for colorbar (0-1).
    """
    gap_width = int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))
    gap = np.full((data_ref.shape[0], gap_width), np.nan)
    combined = np.hstack([data_ref, gap, data_comp])

    _plot_depthmap_on_axes(ax, fig, combined, scale, title, shrink_colorbar)


DEFAULT_COLORMAP = "viridis"


def metadata_to_table_data(metadata: dict, wrap_width: int):
    table_data = []
    for k, v in metadata.items():
        wrapped_lines = textwrap.wrap(str(v), width=wrap_width)
        if not wrapped_lines:
            wrapped_lines = [""]

        # First line has the key
        table_data.append([f"{k}:", wrapped_lines[0]])

        # Continuation lines have empty key
        for line in wrapped_lines[1:]:
            table_data.append(["", line])
    return table_data
