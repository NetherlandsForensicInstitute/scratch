from typing import Literal, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from container_models.base import StriationProfile, FloatArray2D, ImageRGB
from plots.utils import (
    side_by_side_gap_width,
    DEFAULT_COLORMAP,
    get_figure_dimensions,
    render_single_panel,
)


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
    colorbar_label: str | None = "Scan Depth [µm]",
    colorbar_width: str = "2.5%",
    colorbar_pad: float = 0.05,
    aspect: Literal["equal", "auto"] = "equal",
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
    :param aspect: Matplotlib aspect argument passed to imshow.
    """
    gap_width = side_by_side_gap_width(data_ref, data_comp)
    gap = np.full((data_ref.shape[0], gap_width), np.nan)
    combined = np.hstack([data_ref, gap, data_comp])

    plot_depth_map_on_axes(
        ax,
        fig,
        combined,
        scale,
        title,
        colorbar_label=colorbar_label,
        colorbar_width=colorbar_width,
        colorbar_pad=colorbar_pad,
        aspect=aspect,
    )


def plot_depth_map_on_axes(
    ax: Axes,
    fig: Figure,
    data: FloatArray2D,
    scale: float,
    title: str,
    colorbar_label: str | None = "Scan Depth [µm]",
    colorbar_width: str = "5%",
    colorbar_pad: float = 0.05,
    aspect: Literal["equal", "auto"] = "equal",
    color_sigma: float = 3.0,
    extendfrac: float = 0.08,
) -> None:
    """
    Plot a depth map on the given axes.

    Colour scaling clips to median ± color_sigma * (1.4826*MAD) so outlier
    pixels don't wash out the contrast. Red lines on the colorbar mark the
    clip bounds; the true data min/max are labelled at the extend triangle tips.

    :param ax: Matplotlib axes to plot on.
    :param fig: Figure (needed for colorbar).
    :param data: Data to plot in meters.
    :param scale: Scale of the data in meters.
    :param title: Title for the plot.
    :param colorbar_label: Label for the colorbar, or None for no label.
    :param colorbar_width: Width of colorbar as percentage of axes.
    :param colorbar_pad: Padding between plot and colorbar.
    :param aspect: Matplotlib aspect argument passed to imshow.
    :param color_sigma: Robust-std multiplier for the clip bounds (red lines).
    :param extendfrac: Fraction of the colorbar length per extend triangle.
    """
    height, width = data.shape
    extent = (0, width * scale * 1e6, 0, height * scale * 1e6)

    data_um = data * 1e6
    vmin, vmax = _robust_color_limits(data_um, k=color_sigma)

    im = ax.imshow(
        data_um,
        cmap=DEFAULT_COLORMAP,
        aspect=aspect,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("X - Position [µm]", fontsize=11)
    ax.set_ylabel("Y - Position [µm]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=10)

    arr = np.ma.masked_invalid(data_um)
    if arr.count():
        true_min, true_max = float(arr.min()), float(arr.max())
    else:
        true_min, true_max = vmin, vmax

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_width, pad=colorbar_pad)
    cbar = fig.colorbar(
        im, cax=cax, label=colorbar_label, extend="both", extendfrac=extendfrac
    )
    cbar.ax.tick_params(labelsize=10)

    # Red lines at the actual clip bounds.
    cbar.ax.axhline(vmin, color="red", linewidth=2)
    cbar.ax.axhline(vmax, color="red", linewidth=2)

    # Drop regular ticks that would crowd the red lines / tip labels.
    margin = 0.06 * (vmax - vmin)
    cbar.set_ticks([t for t in cbar.get_ticks() if vmin + margin <= t <= vmax - margin])

    # True min/max at the triangle tips. Anchor to the colorbar's right edge
    # (transAxes x=1.0) with a fixed offset in points, matching how matplotlib
    # places the regular tick labels — an offset in axes fraction would scale
    # with the colorbar width and drift between subplots.
    tick_pad_pt = (
        cbar.ax.yaxis.majorTicks[0].get_pad() if cbar.ax.yaxis.majorTicks else 3.5
    )
    if tick_pad_pt is None:
        tick_pad_pt = 3.5
    tip_kw: dict[str, Any] = dict(
        xycoords=cbar.ax.transAxes,
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        annotation_clip=False,
    )
    cbar.ax.annotate(
        f"{true_max:.2f}", xy=(1.0, 1.0 + extendfrac), xytext=(tick_pad_pt, 0), **tip_kw
    )
    cbar.ax.annotate(
        f"{true_min:.2f}", xy=(1.0, -extendfrac), xytext=(tick_pad_pt, 0), **tip_kw
    )


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

    return render_single_panel(
        (fig_width, fig_height),
        lambda fig, ax: plot_depth_map_on_axes(ax, fig, data, scale, title),
    )


def _robust_color_limits(
    data: FloatArray2D,
    k: float = 3.0,
) -> tuple[float, float]:
    """
    Compute color scale limits that exclude outliers via robust sigma clipping.

    Uses median ± k * (1.4826 * MAD) rather than mean ± k*std. The MAD
    (median absolute deviation) is not inflated by extreme outliers the way
    the ordinary standard deviation is, so a heavy-tailed surface (a few very
    deep/high pixels) still yields a tight, high-contrast clip. The 1.4826
    factor rescales MAD to match std for normally distributed data, so `k`
    keeps its usual "~k sigma" meaning. The bound is symmetric about the
    median by construction, even when the data is skewed.

    :param data: Surface data (may contain NaNs for masked/invalid pixels).
    :param k: Number of (robust) standard deviations to clip at.
    :return: (vmin, vmax) tuple in the same units as `data`.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0

    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))

    if mad > 0.0:
        scale = 1.4826 * mad
    else:
        # MAD degenerate (e.g. >50% identical values); fall back to std.
        scale = float(np.std(finite))
        if scale == 0.0:
            return center - 1.0, center + 1.0

    return center - k * scale, center + k * scale


def _plot_surface_with_colorbar(
    fig: Figure,
    ax: Axes,
    im: AxesImage,
    title: str,
    color_sigma: float = 3.0,
    extendfrac: float = 0.08,
) -> None:
    """
    Plot a cell overlay on axes and add an outlier-aware colorbar.

    The image is (re)clipped to median ± color_sigma * (1.4826*MAD) so the
    surface colours and the colorbar agree: the main body spans the clipped range,
    red lines mark that clipping boundary, and the true (unclipped) data
    min/max appear at the tips of the extend triangles. Everything is read
    back off ``im`` itself, so no data array needs to be passed in.

    :param fig: Figure to attach the colorbar to.
    :param ax: Axes the image was plotted on.
    :param im: AxesImage returned by plot_cell_overlay_on_axes.
    :param title: Axes title.
    :param color_sigma: MAD multiplier for the clip bounds (red lines).
    :param extendfrac: Fraction of the colorbar length per extend triangle.
    """
    ax.set_title(title, fontsize=12, fontweight="bold")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    data = im.get_array()
    if data is None:
        raise ValueError("image has no data")
    arr = np.ma.masked_invalid(data)

    # Clip bounds (median ± k*MAD), applied to the image so the *surface*
    # colours are clipped too — this is what puts the red line at ~1.27
    # instead of at the true max. If the image is already clipped upstream
    # this is effectively a no-op, so it's safe to keep either way.
    vmin, vmax = _robust_color_limits(arr.filled(np.nan), k=color_sigma)
    im.set_clim(vmin, vmax)

    if arr.count():
        true_min, true_max = float(arr.min()), float(arr.max())
    else:
        true_min, true_max = vmin, vmax

    cbar = fig.colorbar(
        im, cax=cax, label="Scan Depth [µm]", extend="both", extendfrac=extendfrac
    )
    cbar.ax.tick_params(labelsize=9)

    # Red lines at the actual clip bounds (now ~±1.27, not the true max).
    cbar.ax.axhline(vmin, color="red", linewidth=2)
    cbar.ax.axhline(vmax, color="red", linewidth=2)

    # Drop regular ticks that would crowd the red lines / tip labels.
    margin = 0.06 * (vmax - vmin)
    cbar.set_ticks([t for t in cbar.get_ticks() if vmin + margin <= t <= vmax - margin])

    # True min/max at the triangle tips. Anchor to the colorbar's right edge
    # (transAxes x=1.0) with a fixed offset in points, matching how matplotlib
    # places the regular tick labels — an offset in axes fraction would scale
    # with the colorbar width and drift between subplots.
    tick_pad_pt = (
        cbar.ax.yaxis.majorTicks[0].get_pad() if cbar.ax.yaxis.majorTicks else 3.5
    )
    if tick_pad_pt is None:
        tick_pad_pt = 3.5
    tip_kw: dict[str, Any] = dict(
        xycoords=cbar.ax.transAxes,
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        annotation_clip=False,
    )
    cbar.ax.annotate(
        f"{true_max:.2f}", xy=(1.0, 1.0 + extendfrac), xytext=(tick_pad_pt, 0), **tip_kw
    )
    cbar.ax.annotate(
        f"{true_min:.2f}", xy=(1.0, -extendfrac), xytext=(tick_pad_pt, 0), **tip_kw
    )
