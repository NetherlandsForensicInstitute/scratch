from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from container_models.base import FloatArray2D
from conversion.surface_comparison.cell_registration.utils import (
    REJECTED_SCORE,
    _prepare_rotated_batch,
    _prepare_templates,
    canvas_to_image,
    image_to_canvas,
    iter_score_maps,
    paired_score_maps,
    rotated_crop,
    rotated_shape,
    batched_match,
)

logger = logging.getLogger(__name__)

DEFAULT_REDUCTION = 6
DEFAULT_N_CANDIDATES = 3
DEFAULT_JOBS_PER_CHUNK = 256
#: A coarse pixel with less than this fraction of valid sub-pixels is treated as missing.
_COARSE_VALIDITY_THRESHOLD = 0.5
#: Smallest coarse cell, per side, that can still localise reliably. Below roughly this the coarse
#: stage has too few pixels to discriminate and silently returns the wrong pose - measured on a
#: large-rotation case, a 4x4 coarse cell picked the wrong angle entirely while 5x5 did not.
#: ``reduction`` is capped so this holds, because the failure is silent and the caller cannot be
#: expected to re-derive it for every cell size.
_MIN_COARSE_CELL = 8


def effective_reduction(cell_shape: tuple[int, int], reduction: int) -> int:
    """
    Cap *reduction* so the coarse cell keeps at least :data:`_MIN_COARSE_CELL` pixels per side.

    :param cell_shape: ``(cell_height, cell_width)`` at full resolution.
    :param reduction: Requested reduction factor.
    :returns: The reduction actually usable for this cell size, at least 1.
    """
    return max(1, min(reduction, min(cell_shape) // _MIN_COARSE_CELL))


def downsample(image: FloatArray2D, factor: int) -> FloatArray2D:
    """
    Reduce *image* by *factor* using area averaging, propagating missing data correctly.

    ``cv2.INTER_AREA`` averages over each source block, so NaN anywhere in a block would poison the
    whole output pixel. Instead the valid pixels are averaged among themselves, and a coarse pixel
    is marked missing only when most of its source block was missing.

    :param image: Input 2D array, NaN where data is missing.
    :param factor: Integer reduction factor.
    :returns: Float64 array of shape ``(ceil(height / factor), ceil(width / factor))``.
    """
    height, width = image.shape
    size = (int(np.ceil(width / factor)), int(np.ceil(height / factor)))

    valid = np.isfinite(image)
    filled = np.where(valid, image, 0.0).astype(np.float32)

    mean_of_filled = cv2.resize(filled, size, interpolation=cv2.INTER_AREA)
    mean_of_valid = cv2.resize(
        valid.astype(np.float32), size, interpolation=cv2.INTER_AREA
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        result = mean_of_filled / mean_of_valid
    result[mean_of_valid < _COARSE_VALIDITY_THRESHOLD] = np.nan
    return np.asarray(result, dtype=np.float64)


def _to_full_resolution(coarse_value: float, factor: int) -> float:
    """Map a coordinate in a block-averaged image back to the original grid."""
    return coarse_value * factor + (factor - 1) / 2.0


def _top_candidates(
    scores: torch.Tensor, n_candidates: int, suppression_radius: int
) -> list[list[tuple[int, int, int]]]:
    """
    Extract the strongest well-separated peaks per template from a score volume.

    A candidate is a *location*, so the angle axis is reduced away first and each location competes
    only against other locations. After a peak is taken its neighbourhood is suppressed, otherwise
    the candidates would all be one blurred maximum.

    Only the location is kept, not its angle. At useful reduction factors the area averaging blurs
    out the detail that discriminates rotation, so the coarse angle ranking is not informative:
    trusting it dropped exact agreement from 12/12 to 5/12 in testing. Refinement therefore retries
    the whole sweep at each location. The coarse stage decides *where*, not *at what angle*.

    :param scores: ``(n_angles, n_templates, height, width)``.
    :param n_candidates: Peaks to keep per template.
    :param suppression_radius: Half-width of the suppressed neighbourhood, in pixels.
    :returns: Per template, a list of ``(x, y, angle_index)`` ordered by score, where the angle
        index is used only to map the location back to image coordinates.
    """
    best_over_angles, best_angle = scores.max(dim=0)
    n_templates, _, width = best_over_angles.shape

    results = []
    for index in range(n_templates):
        surface = best_over_angles[index].clone()
        angles = best_angle[index]
        found: list[tuple[int, int, int]] = []
        for _ in range(n_candidates):
            position = int(surface.argmax())
            y, x = divmod(position, width)
            if float(surface[y, x]) <= REJECTED_SCORE:
                break
            found.append((x, y, int(angles[y, x])))
            surface[
                max(0, y - suppression_radius) : y + suppression_radius + 1,
                max(0, x - suppression_radius) : x + suppression_radius + 1,
            ] = -np.inf
        results.append(found)
    return results


def _refine(
    image: FloatArray2D,
    templates: torch.Tensor,
    jobs: list[tuple[int, float, float, float]],
    margin: int,
    minimum_fill_fraction: float,
    fill_value: float,
    standardisation: tuple[float, float],
    jobs_per_chunk: int,
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
    :param jobs_per_chunk: Jobs scored per batch.
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

    for start in range(0, len(jobs), jobs_per_chunk):
        block = jobs[start : start + jobs_per_chunk]
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
    image: FloatArray2D,
    templates: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value: float,
    reduction: int = DEFAULT_REDUCTION,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    margin: int | None = None,
    jobs_per_chunk: int = DEFAULT_JOBS_PER_CHUNK,
    device: torch.device | None = None,
) -> list[tuple[float, int, int, float]]:
    """
    Two-stage search: exhaustive at reduced resolution, then local at full resolution.

    Drop-in replacement for :func:`batched_match` with the same return contract. The angle sweep
    stays global at both stages, so grossly misoriented marks are still found; only the
    *translation* search becomes local, and only once a candidate location has been found.

    :param image: Padded comparison image, NaN outside the original data.
    :param templates: Reference cell data, all the same shape and free of NaN.
    :param angles: Angle sweep in degrees.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value: Value substituted for NaN in the comparison image.
    :param reduction: Coarse-stage reduction factor.
    :param n_candidates: Coarse peaks retained per cell for refinement.
    :param margin: Refinement search radius in pixels; defaults to ``2 * reduction``.
    :param jobs_per_chunk: Candidate poses scored per refinement batch.
    :param device: Torch device; defaults to CUDA when available.
    :returns: Per template, ``(score, x, y, angle_deg)`` with ``x``/``y`` in rotated-canvas pixels.
    """
    if not templates:
        return []
    cell_shape = templates[0].shape
    if any(template.shape != cell_shape for template in templates):
        raise ValueError("All templates must have the same shape.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    usable = effective_reduction(cell_shape, reduction)
    if usable != reduction:
        logger.info(
            "Reduction capped from %d to %d: cells are %dx%d px, and a coarse cell below %d px "
            "per side cannot localise reliably.",
            reduction,
            usable,
            cell_shape[1],
            cell_shape[0],
            _MIN_COARSE_CELL,
        )
    if usable < 2:
        # Nothing left to gain; the coarse stage would be full resolution.
        return batched_match(
            image, templates, angles, minimum_fill_fraction, fill_value, device=device
        )
    reduction = usable
    if margin is None:
        margin = 2 * reduction

    angles = np.asarray(angles, dtype=np.float64)
    sorted_angles = angles[np.lexsort((angles, np.abs(angles)))]
    default_angle = float(sorted_angles[0])

    # coarse stage: find candidate locations
    coarse_image = downsample(image, reduction)
    coarse_templates = [downsample(template, reduction) for template in templates]
    coarse_shape = coarse_templates[0].shape
    if min(coarse_shape) < _MIN_COARSE_CELL:
        logger.warning(
            "Coarse cells are %dx%d px at reduction %d; below roughly %d px the coarse stage "
            "localises poorly. Consider a smaller reduction for this cell size.",
            coarse_shape[1],
            coarse_shape[0],
            reduction,
            _MIN_COARSE_CELL,
        )
    coarse_canvas = (
        max(rotated_shape(*coarse_image.shape, float(a))[0] for a in sorted_angles),
        max(rotated_shape(*coarse_image.shape, float(a))[1] for a in sorted_angles),
    )
    coarse_batch, coarse_valid = _prepare_rotated_batch(
        coarse_image, sorted_angles, fill_value, coarse_canvas
    )
    coarse_tensor, is_non_constant = _prepare_templates(coarse_templates, device)

    candidates: list[list[tuple[int, int, int]]] = []
    for _, scores in iter_score_maps(
        torch.from_numpy(coarse_batch).to(device),
        torch.from_numpy(coarse_valid).to(device),
        coarse_tensor,
        minimum_fill_fraction,
        max(1, min(len(templates), 8)),
        (float(np.nanmean(coarse_image)), float(np.nanstd(coarse_image))),
    ):
        candidates.extend(_top_candidates(scores, n_candidates, max(coarse_shape) // 2))
        del scores
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # refinement stage: full resolution
    jobs: list[tuple[int, float, float, float]] = []
    unusable: list[int] = []
    for index in range(len(templates)):
        if not is_non_constant[index] or not candidates[index]:
            unusable.append(index)
            continue
        for x, y, angle_index in candidates[index]:
            center_x, center_y = canvas_to_image(
                x,
                y,
                coarse_shape,
                coarse_image.shape,
                float(sorted_angles[angle_index]),
            )
            full_x = _to_full_resolution(center_x, reduction)
            full_y = _to_full_resolution(center_y, reduction)
            jobs.extend((index, full_x, full_y, float(a)) for a in sorted_angles)

    fine_tensor, _ = _prepare_templates(templates, device)
    results = _refine(
        image.astype(np.float32),
        fine_tensor,
        jobs,
        margin,
        minimum_fill_fraction,
        fill_value,
        (float(np.nanmean(image)), float(np.nanstd(image))),
        jobs_per_chunk,
        default_angle,
    )
    for index in unusable:
        # A constant reference cell has no defined correlation.
        results[index] = (-1.0, 0, 0, default_angle)

    logger.debug(
        "Coarse-to-fine: reduction %d, %d candidates, margin %d px, %d refinement jobs.",
        reduction,
        n_candidates,
        margin,
        len(jobs),
    )
    return results
