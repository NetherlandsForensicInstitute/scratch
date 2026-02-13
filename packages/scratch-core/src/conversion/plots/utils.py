import textwrap
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from container_models.base import FloatArray2D, ImageRGB, StriationProfile

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


def plot_profiles_on_axes(
    ax: Axes,
    profile_reference: StriationProfile,
    profile_compared: StriationProfile,
    scale: float,
    score: float,
    title: str,
) -> None:
    """
    Plot two aligned profiles on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param profile_reference: Reference profile (aligned, 2D).
    :param profile_compared: Compared profile (aligned, 2D).
    :param scale: scale of the profiles in meters.
    :param score: Pre-computed correlation coefficient.
    :param title: Prefix for the title before the correlation value.
    """
    x1 = np.arange(len(profile_reference)) * scale * 1e6
    x2 = np.arange(len(profile_compared)) * scale * 1e6

    y1 = profile_reference * 1e6
    y2 = profile_compared * 1e6

    ax.plot(x1, y1, "b-", label="Reference Profile A", linewidth=1.5)
    ax.plot(x2, y2, "r-", label="Compared Profile B", linewidth=1.5)

    ax.set_xlabel("Profile Length [µm]", fontsize=11)
    ax.set_ylabel("Profile Height [µm]", fontsize=11)
    ax.set_title(f"{title}: {score:.5f}", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_side_by_side_on_axes(
    ax: Axes,
    fig: Figure,
    data_ref: FloatArray2D,
    data_comp: FloatArray2D,
    scale: float,
    title: str = "Reference Surface A / Moved Compared Surface B",
    colorbar_width: str = "2.5%",
    colorbar_pad: float = 0.05,
) -> None:
    """
    Plot two surfaces side by side on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data_ref: Reference data in meters.
    :param data_comp: Compared data in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param colorbar_width: Width of colorbar as percentage of axes.
    :param colorbar_pad: Padding between plot and colorbar.
    """
    gap_width = int(np.ceil(min(data_ref.shape[1], data_comp.shape[1]) / 100))
    gap = np.full((data_ref.shape[0], gap_width), np.nan)
    combined = np.hstack([data_ref, gap, data_comp])

    plot_depth_map_on_axes(
        ax,
        fig,
        combined,
        scale,
        title,
        colorbar_width=colorbar_width,
        colorbar_pad=colorbar_pad,
    )


def plot_depth_map_on_axes(
    ax: Axes,
    fig: Figure,
    data: FloatArray2D,
    scale: float,
    title: str,
    colorbar_width: str = "5%",
    colorbar_pad: float = 0.05,
) -> None:
    """
    Plot a depth map on the given axes.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data: Data to plot in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param colorbar_width: Width of colorbar as percentage of axes.
    :param colorbar_pad: Padding between plot and colorbar.
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

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
    cbar = fig.colorbar(im, cax=cax, label="Scan Depth [µm]")
    cbar.ax.tick_params(labelsize=10)


def metadata_to_table_data(
    metadata: dict[str, str], wrap_width: int
) -> list[list[str]]:
    """
    Convert metadata dictionary to table rows with text wrapping.

    Long values are wrapped across multiple rows, with the key only
    appearing on the first row.

    :param metadata: Dictionary of metadata key-value pairs.
    :param wrap_width: Maximum character width before wrapping values.
    :returns: Table rows as list of [key, value] string pairs.
    """
    table_data: list[list[str]] = []
    for k, v in metadata.items():
        wrapped_lines = textwrap.wrap(str(v), width=wrap_width)
        if not wrapped_lines:
            wrapped_lines = [""]

        table_data.append([f"{k}:" if k else "", wrapped_lines[0]])

        for line in wrapped_lines[1:]:
            table_data.append(["", line])
    return table_data


def _calculate_text_height(
    text: str, wrap_width: int, line_height: float = 1.5
) -> float:
    """Calculate the number of lines needed for wrapped text."""
    if not text:
        return line_height
    wrapped = textwrap.wrap(str(text), width=wrap_width)
    return max(1, len(wrapped)) * line_height


def _calculate_table_rows(metadata: dict, wrap_width: int = 25) -> int:
    """Calculate total number of display rows including wrapped lines."""
    total_rows = 0
    for key, value in metadata.items():
        key_lines = len(textwrap.wrap(f"{key}:", width=wrap_width))
        value_lines = len(textwrap.wrap(str(value), width=wrap_width))
        total_rows += max(key_lines, value_lines)
    return total_rows


def get_col_widths(
    side_margin: float,
    table_data: list[list[str]],
) -> tuple[float, float]:
    """
    Calculate column widths for a two-column table based on content length.

    The key column width is proportional to the longest key relative to the
    longest value, clamped between 35% and 50% of the available width.

    :param side_margin: Margin on each side as a fraction of total width (0-0.5).
    :param table_data: List of (key, value) string pairs representing table rows.
    :returns: Tuple of (key_column_width, value_column_width) as fractions of
        total width, accounting for side margins.
    """
    available_width = 1.0 - 2 * side_margin

    max_key_len = max(len(row[0]) for row in table_data)
    max_val_len = max(len(row[1]) for row in table_data)
    total_len = max_key_len + max_val_len

    key_ratio = max(0.35, min(0.5, max_key_len / total_len))
    key_width = key_ratio * available_width
    val_width = (1.0 - key_ratio) * available_width
    return key_width, val_width


def get_bounding_box(side_margin: float, table_data: list[list[str]]) -> Bbox:
    """
    Calculate bounding box dimensions for a table with adaptive row heights.

    Row height adapts to content: fewer rows get more generous spacing,
    while many rows use compact spacing to fit. The table is vertically
    centered within the available space.

    :param side_margin: Margin on each side as a fraction of total width (0-0.5).
    :param table_data: List of rows, where each row is a list of cell strings.
    :returns: Bounding box with (left, bottom, width, height) as fractions
        suitable for use as a matplotlib table bbox parameter.
    """
    n_rows = len(table_data)
    available_width = 1.0 - 2 * side_margin

    # Adaptive row height - more rows = tighter spacing, fewer rows = more space
    if n_rows <= 5:
        row_height_fraction = 0.14
    elif n_rows <= 8:
        row_height_fraction = 0.10
    else:
        row_height_fraction = 0.07

    table_height = min(0.92, n_rows * row_height_fraction)
    table_height = max(table_height, 0.5)
    bottom = (1.0 - table_height) / 2

    return Bbox.from_bounds(side_margin, bottom, available_width, table_height)


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


def get_metadata_dimensions(
    metadata_compared: dict, metadata_reference: dict, wrap_width: int
) -> tuple[int, float]:
    """
    Calculate metadata section dimensions based on content.

    Determines the number of display rows needed for the larger of the two
    metadata dictionaries (accounting for text wrapping), and computes an
    appropriate height ratio with a minimum to ensure readability.

    :param metadata_compared: Metadata dictionary for the compared profile.
    :param metadata_reference: Metadata dictionary for the reference profile.
    :param wrap_width: Maximum characters per line before wrapping.
    :returns: Tuple of (max_metadata_rows, metadata_height_ratio) where
        max_metadata_rows is the number of wrapped text rows and
        metadata_height_ratio is the relative height for the metadata row.
    """
    # Calculate content-based heights
    meta_reference_rows = _calculate_table_rows(metadata_reference, wrap_width)
    meta_compared_rows = _calculate_table_rows(metadata_compared, wrap_width)

    # Row 0: based on max metadata content (with minimum for readability)
    max_metadata_rows = max(meta_reference_rows, meta_compared_rows)
    metadata_height_ratio = max(
        0.12, max_metadata_rows * 0.022
    )  # Increased minimum and scale
    return max_metadata_rows, metadata_height_ratio


def plot_depth_map_with_axes(
    data: FloatArray2D,
    scale: float,
    title: str,
) -> ImageRGB:
    """
    Plot a depth map rendering of a mark.

    :param data: data to plot in meters.
    :param scale: scale of the data in meters.
    :param title: Title for the plot.
    :returns: RGB image as uint8 array with shape (H, W, 3).
    """
    height, width = data.shape
    fig_height, fig_width = get_figure_dimensions(height, width)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_depth_map_on_axes(ax, fig, data, scale, title)

    fig.tight_layout()
    arr = figure_to_array(fig)
    plt.close(fig)
    return arr


def draw_metadata_box(
    ax: Axes,
    metadata: dict[str, str],
    title: str | None = None,
    draw_border: bool = True,
    wrap_width: int = 25,
    side_margin: float = 0.06,
) -> None:
    """Draw a metadata box with key-value pairs."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(draw_border)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    table_data = metadata_to_table_data(metadata, wrap_width=wrap_width)
    col_widths = get_col_widths(side_margin, table_data)
    bounding_box = get_bounding_box(side_margin, table_data)

    table = ax.table(
        cellText=table_data,
        cellLoc="left",
        colWidths=col_widths,
        loc="upper center",
        edges="open",
        bbox=bounding_box,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i in range(len(table_data)):
        table[i, 0].set_text_props(fontweight="bold", ha="right")
        table[i, 0].PAD = 0.02
        table[i, 1].set_text_props(ha="left")
        table[i, 1].PAD = 0.02
