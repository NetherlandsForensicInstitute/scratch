"""Fine stage: score candidate poses at full resolution, on small crops around each prediction."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from container_models.base import FloatArray2D
from conversion.surface_comparison.cell_registration.geometry import (
    crop_rotated_image,
    map_image_to_canvas,
)
from conversion.surface_comparison.cell_registration.models import Match, RefinementJob
from conversion.surface_comparison.cell_registration.scoring import (
    clamp_score,
    compute_paired_score_maps,
)


class CropWindow(NamedTuple):
    """A job's search window, placed on the rotated canvas."""

    job: RefinementJob
    left: int
    top: int


def refine(
    image: FloatArray2D,
    templates: torch.Tensor,
    jobs: list[RefinementJob],
    margin: int,
    minimum_fill_fraction: float,
    fill_value: float,
    mean_and_std: tuple[float, float],
    batch_size: int,
    default_angle: float,
) -> list[Match]:
    """
    Score every candidate pose at full resolution, batched across all cells at once.

    Each job is scored on a crop of ``cell + 2 * margin`` per side rather than the whole canvas.
    That is where the saving comes from: at a 150px cell in a 2100px canvas it is roughly two
    orders of magnitude less area per evaluation. Batching every job together matters as much as
    the crop, since scoring one cell at a time leaves the work dominated by per-call overhead.

    :param image: Padded comparison image at full resolution, already float32.
    :param templates: Centered unit-norm templates ``(n_templates, 1, cell_height, cell_width)``.
    :param jobs: Candidate poses to score.
    :param margin: Search radius in pixels around each predicted position.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value: Value substituted for NaN.
    :param mean_and_std: Global statistics of the comparison image.
    :param batch_size: Jobs scored per batch.
    :param default_angle: Angle recorded for cells with no job at all.
    :returns: The best :class:`Match` per template.
    """
    device = templates.device
    n_templates, _, cell_height, cell_width = templates.shape
    cell_shape = (cell_height, cell_width)
    crop_shape = (cell_height + 2 * margin, cell_width + 2 * margin)

    best = [Match(-np.inf, 0, 0, default_angle) for _ in range(n_templates)]

    for start in range(0, len(jobs), batch_size):
        windows = [
            place_crop_window(job, cell_shape, image.shape, margin)
            for job in jobs[start : start + batch_size]
        ]
        crops, validities = _cut_crops(image, windows, crop_shape, fill_value)
        indices = torch.tensor(
            [window.job.cell_index for window in windows],
            dtype=torch.long,
            device=device,
        )
        scores = compute_paired_score_maps(
            torch.from_numpy(crops).to(device),
            torch.from_numpy(validities).to(device),
            templates[indices],
            minimum_fill_fraction,
            mean_and_std,
        )
        out_width = scores.shape[3]
        values, positions = scores.reshape(len(windows), -1).max(dim=1)
        del scores

        for window, value, flat in zip(windows, values.tolist(), positions.tolist()):
            index = window.job.cell_index
            if value > best[index].score:
                best[index] = Match(
                    clamp_score(float(value), index),
                    window.left + int(flat % out_width),
                    window.top + int(flat // out_width),
                    float(window.job.angle_deg),
                )
    return best


def place_crop_window(
    job: RefinementJob,
    cell_shape: tuple[int, int],
    image_shape: tuple[int, ...],
    margin: int,
) -> CropWindow:
    """Place one job's search window on the rotated canvas: the single image-to-canvas mapping."""
    left, top = map_image_to_canvas(
        job.center_x,
        job.center_y,
        cell_shape,
        (image_shape[0], image_shape[1]),
        job.angle_deg,
    )
    return CropWindow(job, int(round(left)) - margin, int(round(top)) - margin)


def _cut_crops(
    image: FloatArray2D,
    windows: list[CropWindow],
    crop_shape: tuple[int, int],
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut each window out of the rotated canvas. :returns: ``(crops, validities)``, ``(n, 1, *crop_shape)``."""
    crop_height, crop_width = crop_shape
    crops = np.empty((len(windows), 1, crop_height, crop_width), dtype=np.float32)
    validities = np.empty_like(crops)

    for position, window in enumerate(windows):
        crop = crop_rotated_image(
            image,
            window.job.angle_deg,
            window.left,
            window.top,
            crop_width,
            crop_height,
            fill_value=np.nan,
        )
        finite = np.isfinite(crop)
        crops[position, 0] = np.where(finite, crop, fill_value)
        validities[position, 0] = finite
    return crops, validities
