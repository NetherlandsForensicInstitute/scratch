import textwrap

from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from conversion.data_formats import MarkMetadata


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


def get_metadata_dimensions(
    metadata_compared: MarkMetadata, metadata_reference: MarkMetadata, wrap_width: int
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
    meta_reference_rows = _calculate_table_rows(
        metadata_reference.to_display_dict(), wrap_width
    )
    meta_compared_rows = _calculate_table_rows(
        metadata_compared.to_display_dict(), wrap_width
    )

    # Row 0: based on max metadata content (with minimum for readability)
    max_metadata_rows = max(meta_reference_rows, meta_compared_rows)
    metadata_height_ratio = max(
        0.12, max_metadata_rows * 0.022
    )  # Increased minimum and scale
    return max_metadata_rows, metadata_height_ratio


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


def draw_metadata_pair(
    ax_ref: Axes,
    ax_comp: Axes,
    metadata_reference: MarkMetadata,
    metadata_compared: MarkMetadata,
    noun: str,
    wrap_width: int = 25,
) -> None:
    """Draw the reference/compared metadata boxes titled "Reference {noun} (A)" etc."""
    draw_metadata_box(
        ax_ref,
        metadata_reference.to_display_dict(),
        f"Reference {noun} (A)",
        wrap_width=wrap_width,
    )
    draw_metadata_box(
        ax_comp,
        metadata_compared.to_display_dict(),
        f"Compared {noun} (B)",
        wrap_width=wrap_width,
    )
