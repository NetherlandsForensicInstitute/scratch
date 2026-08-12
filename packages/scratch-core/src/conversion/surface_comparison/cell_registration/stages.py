"""Two-stage search runners: coarse sweep and fine refinement."""

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
    search_candidates,
    sort_by_absolute_angle,
)


def run_coarse_stage(
    image_coarse: FloatArray2D,
    templates_coarse: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value_coarse: float,
    n_candidates: int = 3,
    template_batch_size: int | None = None,
    angle_batch_size: int | None = None,
    device: torch.device | None = None,
) -> list[list[Match]]:
    """
    Exhaustive translation + rotation sweep on the downsampled pair.

    Returns up to *n_candidates* poses per template, ordered by score. A template with no viable
    candidate (constant cell, or every position rejected) gets an empty list.

    :param image_coarse: Padded comparison image downsampled for the coarse stage, NaN outside data.
    :param templates_coarse: Reference cell data at coarse resolution, all same shape, free of NaN.
    :param angles: Angle sweep in degrees.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value_coarse: Value substituted for NaN in *image_coarse*.
    :param n_candidates: Peaks retained per template for refinement.
    :param template_batch_size: Templates correlated per chunk.
    :param angle_batch_size: Angles processed per chunk.
    :param device: Torch device; defaults to CUDA when available.
    :returns: One list of :class:`Match` entries per template. Empty list means no viable candidate.
    """
    if not templates_coarse:
        return []
    device = resolve_device(device)
    candidates_raw, _is_usable = search_candidates(
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
    # search_candidates returns a placeholder Match(-1.0, ...) for unusable templates;
    # convert those back to empty lists so callers use the empty-list convention.
    return [
        found if found and found[0].score >= 0.0 else [] for found in candidates_raw
    ]


def run_fine_stage(
    image_full: FloatArray2D,
    templates_full: list[np.ndarray],
    candidates: list[list[Match]],
    coarse_cell_shape: tuple[int, int],
    coarse_image_shape: tuple[int, ...],
    cap_factor: float,
    angles: np.ndarray,
    position_margin: int,
    angle_margin_degrees: float,
    minimum_fill_fraction: float,
    fill_value_full: float,
    fine_batch_size: int | None = None,
    device: torch.device | None = None,
) -> list[Match]:
    """
    Refine coarse candidates at full resolution.

    Translation and rotation are not independent, so refinement does not re-run the whole sweep:
    it searches a window of ``angle_margin_degrees`` around each *candidate's own* coarse angle,
    together with a ``position_margin``-pixel translation window (translation is effectively free,
    since the sliding-window correlation evaluates every position in one shot).

    :param image_full: Padded comparison image at full resolution, NaN outside the original data.
    :param templates_full: Reference cell data at full resolution, all the same shape, free of NaN.
    :param candidates: Coarse-stage candidates per template. Empty list means no viable candidate.
    :param coarse_cell_shape: ``(height, width)`` of the coarse templates.
    :param coarse_image_shape: Shape of the coarse comparison image.
    :param cap_factor: How many full-resolution pixels one coarse pixel spans (>= 1).
    :param angles: Angle sweep used for the coarse stage (for on-grid angle offsets).
    :param position_margin: Refinement translation search radius, in full-resolution pixels.
    :param angle_margin_degrees: Refinement angle search radius, in degrees.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value_full: Value substituted for NaN in *image_full*.
    :param fine_batch_size: Refinement jobs scored per chunk.
    :param device: Torch device; defaults to CUDA when available.
    :returns: One :class:`Match` per template (best refined candidate). Sentinel
        ``Match(-1.0, 0, 0, default_angle)`` for templates with no candidates.
    """
    if not templates_full:
        return []
    device = resolve_device(device)

    trial_offsets = compute_trial_offsets(angles, angle_margin_degrees)
    jobs = build_refinement_jobs(
        candidates,
        coarse_cell_shape=coarse_cell_shape,
        coarse_image_shape=coarse_image_shape,
        cap_factor=cap_factor,
        trial_offsets=trial_offsets,
    )

    default_angle = float(sort_by_absolute_angle(angles)[0])
    template_tensor, is_non_constant = prepare_templates(templates_full, device)
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
    # Constant reference cells and cells with no coarse candidates get a sentinel match.
    for index in range(len(templates_full)):
        if not is_non_constant[index] or not candidates[index]:
            results[index] = Match(-1.0, 0, 0, default_angle)

    total_candidates = sum(len(c) for c in candidates)
    logger.debug(
        "Fine stage: cap factor {:.2f}, {} total coarse candidates, +/-{} px, +/-{:.1f} deg, "
        "{} refinement jobs.",
        cap_factor,
        total_candidates,
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

    :param angles: Angle sweep used for the coarse stage.
    :param angle_margin_degrees: Refinement angle search radius, in degrees.
    :returns: Array of integer multiples of the coarse sweep step, spanning ±*angle_margin_degrees*.
    """
    step = float(np.min(np.diff(np.sort(angles)))) if len(angles) > 1 else 1.0
    n_steps = int(np.ceil(angle_margin_degrees / step))
    return np.arange(-n_steps, n_steps + 1) * step


def build_refinement_jobs(
    candidates: list[list[Match]],
    coarse_cell_shape: tuple[int, int],
    coarse_image_shape: tuple[int, ...],
    cap_factor: float,
    trial_offsets: np.ndarray,
) -> list[RefinementJob]:
    """
    Expand every coarse candidate into one job per trial angle, at full resolution.

    Empty candidate lists are skipped (no viable coarse match for that cell).

    :param candidates: Coarse-stage candidates per template. Empty list means no viable candidate.
    :param coarse_cell_shape: ``(height, width)`` of the coarse templates.
    :param coarse_image_shape: Shape of the coarse comparison image.
    :param cap_factor: How many full-resolution pixels one coarse pixel spans.
    :param trial_offsets: Angle offsets (degrees) to try around each candidate.
    :returns: List of :class:`RefinementJob` instances, one per candidate-angle combination.
    """
    jobs: list[RefinementJob] = []
    for index, found in enumerate(candidates):
        if not found:
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
