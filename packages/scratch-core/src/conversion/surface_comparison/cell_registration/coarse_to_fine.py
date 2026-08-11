"""Two-stage search: exhaustive sweep on a downsampled image pair, then local refinement."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from container_models.base import FloatArray2D
from conversion.surface_comparison.cell_registration.batching import (
    DEFAULT_FINE_BATCH_SIZE,
    get_default_batch_size,
    resolve_device,
)
from conversion.surface_comparison.cell_registration.geometry import (
    map_canvas_to_image,
    map_coarse_to_full,
)
from conversion.surface_comparison.cell_registration.models import Match, RefinementJob
from conversion.surface_comparison.cell_registration.refine import refine
from conversion.surface_comparison.cell_registration.scoring import (
    compute_mean_and_std,
    prepare_templates,
)
from conversion.surface_comparison.cell_registration.search import (
    find_best_matches,
    get_uniform_cell_shape,
    search_candidates,
    sort_by_absolute_angle,
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
) -> list[Match]:
    """
    Sweep the downsampled pair exhaustively, then refine each candidate at full resolution.

    Drop-in replacement for :func:`~.search.find_best_matches` with the same return contract, built
    from the same primitives: the coarse stage is :func:`~.search.search_candidates` over the full
    angle sweep, and refinement scores the same correlation on small crops around each candidate.

    Translation and rotation are not independent, so refinement does not re-run the whole sweep: it
    searches a window of ``angle_margin_degrees`` around each *candidate's own* coarse angle,
    together with a ``position_margin``-pixel translation window (translation is effectively free,
    since the sliding-window correlation evaluates every position in one shot).

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
    device = resolve_device(device)

    if cap_factor <= 1.0:
        # The coarse stage would search the same resolution as refinement; do one exhaustive pass
        # instead of the same work twice.
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

    candidates, is_usable = search_candidates(
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
    jobs = build_refinement_jobs(
        candidates,
        is_usable,
        coarse_cell_shape=get_uniform_cell_shape(templates_coarse),
        coarse_image_shape=image_coarse.shape,
        cap_factor=cap_factor,
        trial_offsets=compute_trial_offsets(angles, angle_margin_degrees),
    )

    default_angle = float(sort_by_absolute_angle(angles)[0])
    template_tensor, _ = prepare_templates(templates_full, device)
    results = refine(
        image_full.astype(np.float32),
        template_tensor,
        jobs,
        position_margin,
        minimum_fill_fraction,
        fill_value_full,
        compute_mean_and_std(image_full),
        fine_batch_size or get_default_batch_size(device, DEFAULT_FINE_BATCH_SIZE),
        default_angle,
    )
    # A constant reference cell, or one with no viable coarse candidate, has no defined match.
    for index, usable in enumerate(is_usable):
        if not usable:
            results[index] = Match(-1.0, 0, 0, default_angle)

    logger.debug(
        "Coarse-to-fine: cap factor {:.2f}, {} candidates/cell, +/-{} px, +/-{:.1f} deg, "
        "{} refinement jobs.",
        cap_factor,
        n_candidates,
        position_margin,
        angle_margin_degrees,
        len(jobs),
    )
    return results


def compute_trial_offsets(
    angles: np.ndarray, angle_margin_degrees: float
) -> np.ndarray:
    """
    Angle offsets to try around each candidate's coarse angle.

    On the sweep grid rather than in 1-degree steps: the coarse angle is only reliable to about one
    sweep step, and reporting angles off-grid widens the residual distribution that
    ``angle_deviation_threshold`` gates on. Measured on a real pair, mean |residual| for
    partial-fill cells was 5.12 deg on-grid versus 6.00 deg at 1-degree steps, against a 6.0
    threshold - which is what costs the marginal border cells.
    """
    step = float(np.min(np.diff(np.sort(angles)))) if len(angles) > 1 else 1.0
    n_steps = int(np.ceil(angle_margin_degrees / step))
    return np.arange(-n_steps, n_steps + 1) * step


def build_refinement_jobs(
    candidates: list[list[Match]],
    is_usable: list[bool],
    coarse_cell_shape: tuple[int, int],
    coarse_image_shape: tuple[int, ...],
    cap_factor: float,
    trial_offsets: np.ndarray,
) -> list[RefinementJob]:
    """Expand every usable coarse candidate into one job per trial angle, at full resolution."""
    jobs: list[RefinementJob] = []
    for index, found in enumerate(candidates):
        if not is_usable[index]:
            continue
        for match in found:
            center_x, center_y = map_canvas_to_image(
                match.x,
                match.y,
                coarse_cell_shape,
                (coarse_image_shape[0], coarse_image_shape[1]),
                match.angle_deg,
            )
            jobs.extend(
                RefinementJob(
                    index,
                    map_coarse_to_full(center_x, cap_factor),
                    map_coarse_to_full(center_y, cap_factor),
                    float(match.angle_deg + offset),
                )
                for offset in trial_offsets
            )
    return jobs
