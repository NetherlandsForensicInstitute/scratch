"""Shared test helpers for the cell-registration test suite.

All helpers are pure functions and carry no pytest imports so this module is
safe to import from any test file without side-effects.
"""

from __future__ import annotations

import numpy as np
import torch

from container_models.base import DepthData, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.resample import resize_nan_aware
from conversion.surface_comparison.cell_registration.stages import (
    run_coarse_stage,
    run_fine_stage,
)
from conversion.surface_comparison.cell_registration.stage_builders import (
    build_angle_sweep,
    build_coarse_stage,
    build_full_resolution_stage,
    compute_cap_factor,
    convert_grid_cell_to_cell,
    record_results,
)
from conversion.surface_comparison.cell_registration.search import (
    find_best_matches,
    get_uniform_cell_shape,
)
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
    return resize_nan_aware(image, new_shape, interpolation="area")


def identity_params() -> ComparisonParams:
    """Return :class:`ComparisonParams` configured for a zero-angle identity test."""
    return ComparisonParams(
        minimum_fill_fraction=0.5,
        correlation_threshold=0.5,
        search_angle_min=-60.0,
        search_angle_max=60.0,
        search_angle_step=60.0,
    )


def register_cells(
    grid_cells: list[GridCell],
    reference_image: ScanImage,
    comparison_image: ScanImage,
    params: ComparisonParams,
    device: torch.device | None = None,
) -> list[Cell]:
    """
    Register each grid cell by coarse-to-fine template matching (test helper).

    Both images must already be on the same pixel scale.

    :param grid_cells: Reference grid cells; all must share size and ``nan_fill_value``.
    :param reference_image: Reference scan image the grid cells were generated from.
    :param comparison_image: Comparison scan image at the reference's pixel scale.
    :param params: Algorithm parameters.
    :param device: Optional torch device override.
    :returns: One :class:`Cell` per grid cell with its best registration result.
    :raises ValueError: If grid cells disagree on ``nan_fill_value`` or images differ in scale.
    """
    if not grid_cells:
        return []
    if len({grid_cell.nan_fill_value for grid_cell in grid_cells}) > 1:
        raise ValueError("All grid cells must share the same nan_fill_value.")
    if not np.isclose(reference_image.scale_x, comparison_image.scale_x, rtol=1e-6):
        raise ValueError(
            f"Reference ({reference_image.scale_x}) and comparison "
            f"({comparison_image.scale_x}) images must be on the same pixel scale; "
            "resample the comparison image first."
        )

    cell_width, cell_height = grid_cells[0].width, grid_cells[0].height
    cap_factor = compute_cap_factor(
        reference_image, comparison_image, cell_width, cell_height, params.max_size
    )
    full = build_full_resolution_stage(
        comparison_image, grid_cells, cell_width, cell_height
    )
    angles = build_angle_sweep(params)

    if cap_factor > 1.0:
        coarse = build_coarse_stage(
            comparison_image, reference_image, grid_cells, cap_factor
        )
        candidates = run_coarse_stage(
            image_coarse=coarse.image,
            templates_coarse=coarse.templates,
            angles=angles,
            minimum_fill_fraction=params.minimum_fill_fraction,
            fill_value_coarse=coarse.fill_value,
            n_candidates=params.n_candidates,
            template_batch_size=params.template_batch_size,
            angle_batch_size=params.angle_batch_size,
            device=device,
        )
        results = run_fine_stage(
            image_full=full.image,
            templates_full=full.templates,
            candidates=candidates,
            coarse_cell_shape=get_uniform_cell_shape(coarse.templates),
            coarse_image_shape=coarse.image.shape,
            cap_factor=cap_factor,
            angles=angles,
            position_margin=params.fine_n_pixels,
            angle_margin_degrees=params.fine_m_degrees,
            minimum_fill_fraction=params.minimum_fill_fraction,
            fill_value_full=full.fill_value,
            fine_batch_size=params.fine_batch_size,
            device=device,
        )
    else:
        # Coarse and fine would search the same resolution; one exhaustive pass instead of
        # the same work twice. Mirrors conversion.surface_comparison.pipeline.compare_surfaces.
        results = find_best_matches(
            full.image,
            full.templates,
            angles,
            params.minimum_fill_fraction,
            full.fill_value,
            device=device,
            template_batch_size=params.template_batch_size,
            angle_batch_size=params.angle_batch_size,
        )

    record_results(grid_cells, results, full.image.shape, cell_width, cell_height)
    return [
        convert_grid_cell_to_cell(
            grid_cell=grid_cell, pixel_size=reference_image.scale_x
        )
        for grid_cell in grid_cells
    ]


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


def match_coarse_to_fine(
    image_full: FloatArray2D,
    image_coarse: FloatArray2D,
    templates_full: list[np.ndarray],
    templates_coarse: list[np.ndarray],
    cap_factor: float,
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value_full: float,
    fill_value_coarse: float,
    n_candidates: int = 3,
    position_margin: int = 5,
    angle_margin_degrees: float = 5.0,
    template_batch_size: int | None = None,
    angle_batch_size: int | None = None,
    fine_batch_size: int | None = None,
    device: torch.device | None = None,
) -> list:
    """
    Sweep the downsampled pair exhaustively, then refine each candidate at full resolution.

    Test-only convenience wrapper around :func:`run_coarse_stage` and :func:`run_fine_stage`.
    When ``cap_factor <= 1.0``, falls back to a single exhaustive pass.

    :param image_full: Padded comparison image at full resolution, NaN outside the original data.
    :param image_coarse: Padded comparison image downsampled for the coarse stage, same convention.
    :param templates_full: Reference cell data at full resolution, all the same shape, free of NaN.
    :param templates_coarse: The same cells downsampled, free of NaN, aligned 1:1 with *templates_full*.
    :param cap_factor: How many full-resolution pixels one coarse pixel spans (>= 1).
    :param angles: Angle sweep in degrees, used for the coarse stage.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value_full: Value substituted for NaN in *image_full*.
    :param fill_value_coarse: Value substituted for NaN in *image_coarse*.
    :param n_candidates: Coarse peaks retained per cell for refinement.
    :param position_margin: Refinement translation search radius, in full-resolution pixels.
    :param angle_margin_degrees: Refinement angle search radius, in degrees.
    :param template_batch_size: Templates correlated per chunk in the coarse stage.
    :param angle_batch_size: Angles processed per chunk in the coarse stage.
    :param fine_batch_size: Refinement jobs scored per chunk.
    :param device: Torch device; defaults to CUDA when available.
    :returns: The best :class:`Match` per template, in full-resolution rotated-canvas pixels.
    """
    if not templates_full:
        return []
    if len(templates_full) != len(templates_coarse):
        raise ValueError("templates_full and templates_coarse must be aligned 1:1.")
    get_uniform_cell_shape(templates_full)

    if cap_factor <= 1.0:
        from conversion.surface_comparison.cell_registration.search import (
            find_best_matches,
        )

        return find_best_matches(
            image_full,
            templates_full,
            angles,
            minimum_fill_fraction,
            fill_value_full,
            device=device,
            template_batch_size=template_batch_size,
            angle_batch_size=angle_batch_size,
        )

    candidates = run_coarse_stage(
        image_coarse,
        templates_coarse,
        angles,
        minimum_fill_fraction,
        fill_value_coarse,
        n_candidates=n_candidates,
        template_batch_size=template_batch_size,
        angle_batch_size=angle_batch_size,
        device=device,
    )

    return run_fine_stage(
        image_full,
        templates_full,
        candidates,
        coarse_cell_shape=get_uniform_cell_shape(templates_coarse),
        coarse_image_shape=image_coarse.shape,
        cap_factor=cap_factor,
        angles=angles,
        position_margin=position_margin,
        angle_margin_degrees=angle_margin_degrees,
        minimum_fill_fraction=minimum_fill_fraction,
        fill_value_full=fill_value_full,
        fine_batch_size=fine_batch_size,
        device=device,
    )
