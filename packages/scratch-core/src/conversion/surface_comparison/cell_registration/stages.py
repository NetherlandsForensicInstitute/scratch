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
    get_uniform_cell_shape,
    sort_by_absolute_angle,
)


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

    Searches a window of ``angle_margin_degrees`` around each candidate's coarse angle and a
    ``position_margin``-pixel translation window.

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
    :returns: One Match per template (best refined candidate). Sentinel
        ``Match(-1.0, 0, 0, default_angle)`` for templates with no candidates.
    :raises ValueError: If *candidates* and *templates_full* are not aligned 1:1.
    """
    if not templates_full:
        return []
    if len(candidates) != len(templates_full):
        raise ValueError("candidates and templates_full must be aligned 1:1.")
    device = resolve_device(device)

    trial_offsets = compute_trial_offsets(angles, angle_margin_degrees)
    jobs = build_refinement_jobs(
        candidates,
        coarse_cell_shape=coarse_cell_shape,
        coarse_image_shape=coarse_image_shape,
        full_cell_shape=get_uniform_cell_shape(templates_full),
        cap_factor=cap_factor,
        trial_offsets=trial_offsets,
    )

    default_angle = float(sort_by_absolute_angle(angles)[0])
    template_tensor, is_non_constant = prepare_templates(templates_full, device)
    results = refine(
        image=image_full.astype(np.float32),
        templates=template_tensor,
        jobs=jobs,
        margin=position_margin,
        minimum_fill_fraction=minimum_fill_fraction,
        fill_value=fill_value_full,
        mean_and_std=compute_mean_and_std(image_full),
        batch_size=fine_batch_size
        or get_default_batch_size(device, DEFAULT_FINE_BATCH_SIZE),
        default_angle=default_angle,
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

    The offsets are restricted to the coarse sweep grid to maintain consistency and
    reduce residual distribution variance.

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
    full_cell_shape: tuple[int, int],
    cap_factor: float,
    trial_offsets: np.ndarray,
) -> list[RefinementJob]:
    """
    Expand every coarse candidate into one job per trial angle, at full resolution.

    Empty candidate lists are skipped.

    :param candidates: Coarse-stage candidates per template. Empty list means no viable candidate.
    :param coarse_cell_shape: ``(height, width)`` of the coarse templates.
    :param coarse_image_shape: Shape of the coarse comparison image.
    :param full_cell_shape: ``(height, width)`` of the full-resolution templates.
    :param cap_factor: How many full-resolution pixels one coarse pixel spans.
    :param trial_offsets: Angle offsets (degrees) to try around each candidate.
    :returns: List of RefinementJob instances, one per candidate-angle combination.
    """
    # Both canvases are padded by one cell, so the cell shapes are also the paddings.
    coarse_height, coarse_width = coarse_cell_shape
    full_height, full_width = full_cell_shape
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
                    map_coarse_to_full(center_x, cap_factor, coarse_width, full_width),
                    map_coarse_to_full(
                        center_y, cap_factor, coarse_height, full_height
                    ),
                    float(match.angle_deg + offset),
                )
                for offset in trial_offsets
            )
    return jobs
