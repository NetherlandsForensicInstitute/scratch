from __future__ import annotations

import logging

import numpy as np
import torch

from container_models.base import FloatArray2D
from conversion.surface_comparison.cell_registration.utils import (
    DEFAULT_FINE_BATCH_SIZE,
    batched_match,
    canvas_to_image,
    default_batch_size,
    image_to_canvas,
    paired_score_maps,
    rotated_crop,
    search_candidates,
    _prepare_templates,
)

logger = logging.getLogger(__name__)


def _to_full_resolution(coarse_value: float, factor: float) -> float:
    """Map a coordinate in a block-averaged image back to the original grid."""
    return coarse_value * factor + (factor - 1) / 2.0


def _refine(
        image: FloatArray2D,
        templates: torch.Tensor,
        jobs: list[tuple[int, float, float, float]],
        margin: int,
        minimum_fill_fraction: float,
        fill_value: float,
        standardisation: tuple[float, float],
        batch_size: int,
        default_angle: float,
) -> list[tuple[float, int, int, float]]:
    """
    Score every candidate pose at full resolution, batched across all cells at once.

    Each job is one ``(cell, predicted centre, angle)`` triple, scored on a crop of
    ``cell + 2 * margin`` per side rather than the whole canvas. That is where the saving comes
    from: at a 150px cell in a 2100px canvas it is roughly two orders of magnitude less area per
    evaluation. Batching every job together matters as much as the crop, since scoring one cell at
    a time leaves the work dominated by per-call overhead.

    :param image: Padded comparison image at full resolution, already float32.
    :param templates: Centred unit-norm templates ``(n_templates, 1, cell_height, cell_width)``.
    :param jobs: ``(template_index, centre_x, centre_y, angle_deg)`` per candidate pose.
    :param margin: Search radius in pixels around each predicted position.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value: Value substituted for NaN.
    :param standardisation: Global ``(mean, standard_deviation)`` of the comparison image.
    :param batch_size: Jobs scored per batch.
    :param default_angle: Angle recorded for cells with no viable candidate.
    :returns: Per template, ``(score, x, y, angle_deg)`` with ``x``/``y`` on the rotated canvas.
    """
    device = templates.device
    n_templates, _, cell_height, cell_width = templates.shape
    cell_shape = (cell_height, cell_width)
    crop_height, crop_width = cell_height + 2 * margin, cell_width + 2 * margin

    best: list[tuple[float, int, int, float]] = [
        (-np.inf, 0, 0, default_angle) for _ in range(n_templates)
    ]

    for start in range(0, len(jobs), batch_size):
        block = jobs[start: start + batch_size]
        crops = np.empty((len(block), 1, crop_height, crop_width), dtype=np.float32)
        validities = np.empty_like(crops)
        origins = []

        for position, (index, center_x, center_y, angle) in enumerate(block):
            left, top = image_to_canvas(
                center_x, center_y, cell_shape, image.shape, angle
            )
            left, top = int(round(left)) - margin, int(round(top)) - margin
            crop = rotated_crop(
                image, angle, left, top, crop_width, crop_height, fill_value=np.nan
            )
            finite = np.isfinite(crop)
            crops[position, 0] = np.where(finite, crop, fill_value)
            validities[position, 0] = finite
            origins.append((index, left, top, angle))

        indices = torch.tensor(
            [job[0] for job in block], dtype=torch.long, device=device
        )
        scores = paired_score_maps(
            torch.from_numpy(crops).to(device),
            torch.from_numpy(validities).to(device),
            templates[indices],
            minimum_fill_fraction,
            standardisation,
        )
        out_width = scores.shape[3]
        values, positions = scores.reshape(len(block), -1).max(dim=1)
        del scores

        for position, (value, flat) in enumerate(
                zip(values.tolist(), positions.tolist())
        ):
            index, left, top, angle = origins[position]
            if value > best[index][0]:
                best[index] = (
                    min(max(float(value), -1.0), 1.0),
                    left + int(flat % out_width),
                    top + int(flat // out_width),
                    float(angle),
                )
    return best


def coarse_to_fine_match(
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
) -> list[tuple[float, int, int, float]]:
    """
    Two-stage search: exhaustive sweep on a downsampled image pair, then local refinement at full
    resolution.

    Drop-in replacement for :func:`~conversion.surface_comparison.cell_registration.utils.batched_match`
    with the same return contract, built from exactly the same search primitive
    (:func:`~conversion.surface_comparison.cell_registration.utils.search_candidates`): the coarse
    stage calls it for the full angle sweep with ``n_candidates`` peaks per cell, and refinement
    calls the pairwise variant of the same scoring code (:func:`paired_score_maps`) on small crops
    around each candidate.

    Translation and rotation are not independent, so refinement does not re-run the whole angle
    sweep: it searches a local window of ``angle_margin_degrees``, in 1-degree steps, centred on
    each *candidate's own* coarse-stage angle, together with a ``position_margin``-pixel translation
    window around its position (translation is effectively free: the sliding-window correlation
    evaluates every position in that window in one shot).

    :param image_full: Padded comparison image at full resolution, NaN outside the original data.
    :param image_coarse: Padded comparison image downsampled for the coarse stage, same convention.
    :param templates_full: Reference cell data at full resolution, all the same shape, free of NaN.
    :param templates_coarse: The same cells downsampled for the coarse stage, free of NaN, aligned
        1:1 with *templates_full*.
    :param cap_factor: How many full-resolution pixels one coarse pixel spans (>= 1).
    :param angles: Angle sweep in degrees, used for the coarse stage.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value_full: Value substituted for NaN in *image_full*.
    :param fill_value_coarse: Value substituted for NaN in *image_coarse*.
    :param n_candidates: Coarse peaks retained per cell for refinement.
    :param position_margin: Refinement translation search radius, in full-resolution pixels.
    :param angle_margin_degrees: Refinement angle search radius, in degrees (1-degree steps).
    :param template_batch_size: Templates correlated per chunk in the coarse stage. ``None`` picks
        a device default.
    :param angle_batch_size: Angles processed per chunk in the coarse stage. ``None`` picks a
        device default.
    :param fine_batch_size: Refinement jobs scored per chunk. ``None`` picks a device default.
    :param device: Torch device; defaults to CUDA when available.
    :returns: Per template, ``(score, x, y, angle_deg)`` with ``x``/``y`` in full-resolution
        rotated-canvas pixels.
    """
    if not templates_full:
        return []
    if len(templates_full) != len(templates_coarse):
        raise ValueError("templates_full and templates_coarse must be aligned 1:1.")
    cell_shape = templates_full[0].shape
    if any(template.shape != cell_shape for template in templates_full):
        raise ValueError("All full-resolution templates must have the same shape.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cap_factor <= 1.0:
        # The coarse stage would search the same resolution as refinement; skip straight to one
        # exhaustive pass instead of doing the same work twice.
        return batched_match(
            image_full,
            templates_full,
            angles,
            minimum_fill_fraction,
            fill_value_full,
            device=device,
            template_batch_size=template_batch_size,
            angle_batch_size=angle_batch_size,
        )

    coarse_shape = templates_coarse[0].shape
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

    angles = np.asarray(angles, dtype=np.float64)
    default_angle = float(np.sort(angles)[0])

    jobs: list[tuple[int, float, float, float]] = []
    unusable: list[int] = []
    trial_offsets = np.arange(
        -angle_margin_degrees, angle_margin_degrees + 1.0, 1.0
    )
    for index in range(len(templates_full)):
        if not is_usable[index]:
            unusable.append(index)
            continue
        for _score, x, y, angle in candidates[index]:
            center_x, center_y = canvas_to_image(
                x, y, coarse_shape, image_coarse.shape, angle
            )
            full_x = _to_full_resolution(center_x, cap_factor)
            full_y = _to_full_resolution(center_y, cap_factor)
            jobs.extend(
                (index, full_x, full_y, float(angle + offset)) for offset in trial_offsets
            )

    fine_batch_size = fine_batch_size or default_batch_size(device, DEFAULT_FINE_BATCH_SIZE)
    fine_tensor, _ = _prepare_templates(templates_full, device)
    results = _refine(
        image_full.astype(np.float32),
        fine_tensor,
        jobs,
        position_margin,
        minimum_fill_fraction,
        fill_value_full,
        (float(np.nanmean(image_full)), float(np.nanstd(image_full))),
        fine_batch_size,
        default_angle,
    )
    for index in unusable:
        # A constant reference cell, or one with no viable coarse candidate, has no defined match.
        results[index] = (-1.0, 0, 0, default_angle)

    logger.debug(
        "Coarse-to-fine: cap factor %.2f, %d candidates/cell, +/-%d px, +/-%.1f deg, %d refinement jobs.",
        cap_factor,
        n_candidates,
        position_margin,
        angle_margin_degrees,
        len(jobs),
    )
    return results
