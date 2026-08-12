"""Exhaustive translation + rotation sweep: the one "search everywhere" implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from container_models.base import FloatArray2D
from conversion.surface_comparison.cell_registration.batching import (
    DEFAULT_ANGLE_BATCH_SIZE,
    DEFAULT_TEMPLATE_BATCH_SIZE,
    get_default_batch_size,
    resolve_device,
)
from conversion.surface_comparison.cell_registration.geometry import (
    compute_rotated_shape,
    rotate_image,
)
from conversion.surface_comparison.cell_registration.models import Match
from conversion.surface_comparison.cell_registration.scoring import (
    REJECTED_SCORE,
    clamp_score,
    compute_mean_and_std,
    find_next_fast_length,
    iterate_score_maps,
    precompute_template_ffts,
    prepare_templates,
)


def search_candidates(
    image: FloatArray2D,
    templates: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value: float,
    n_candidates: int = 1,
    suppression_radius: int | None = None,
    template_batch_size: int | None = None,
    angle_batch_size: int | None = None,
    device: torch.device | None = None,
) -> list[list[Match]]:
    """
    Exhaustive translation + rotation sweep; the top *n_candidates* poses per template.

    Angles are processed in order of increasing ``|angle|`` and in chunks of *angle_batch_size*, so
    that: (a) rotated canvases within a chunk are close in size (less of the batch is padding), and
    (b) ties resolve to the smallest ``|angle|``, then the smallest ``y``, then the smallest ``x``.

    :param image: Padded comparison image, NaN outside the original data.
    :param templates: Reference cell data, all the same shape and free of NaN.
    :param angles: Angle sweep in degrees.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value: Value substituted for NaN in the comparison image.
    :param n_candidates: Number of well-separated peaks to keep per template.
    :param suppression_radius: Half-width, in pixels, of the neighborhood suppressed around each
        accepted peak before looking for the next one. Defaults to half the cell's longer side.
    :param template_batch_size: Templates correlated per chunk. ``None`` picks a device default.
    :param angle_batch_size: Angles processed per chunk. ``None`` picks a device default.
    :param device: Torch device; defaults to CUDA when available.
    :returns: Per template, up to *n_candidates* :class:`Match` entries ordered by score. A
        template with no viable candidate — a constant cell, or every position rejected — gets an
        empty list. Scores are genuine Pearson correlations and may be negative, so emptiness is
        the only signal of "no candidate"; never infer it from the score.
    """
    if not templates:
        return []
    cell_height, cell_width = get_uniform_cell_shape(templates)
    device = resolve_device(device)
    angle_batch_size = angle_batch_size or get_default_batch_size(
        device, DEFAULT_ANGLE_BATCH_SIZE
    )
    template_batch_size = template_batch_size or get_default_batch_size(
        device, DEFAULT_TEMPLATE_BATCH_SIZE
    )
    if suppression_radius is None:
        suppression_radius = max(cell_height, cell_width) // 2

    angles = sort_by_absolute_angle(angles)
    canvas_shape = _compute_common_canvas_shape(image.shape, angles)
    out_shape = (canvas_shape[0] - cell_height + 1, canvas_shape[1] - cell_width + 1)

    template_tensor, is_non_constant = prepare_templates(templates, device)
    mean_and_std = compute_mean_and_std(image)
    # The transform size is fixed for the whole sweep (canvas_shape is), so the template spectra
    # are computed once here rather than once per angle chunk.
    template_ffts = precompute_template_ffts(
        template_tensor,
        (
            find_next_fast_length(canvas_shape[0] + cell_height - 1),
            find_next_fast_length(canvas_shape[1] + cell_width - 1),
        ),
        template_batch_size,
    )
    best = _BestPoseGrid.create_empty(len(templates), out_shape, device)

    logger.debug(
        "Matching {} templates over {} angles on {} (chunks: {} angles x {} templates).",
        len(templates),
        len(angles),
        device,
        angle_batch_size,
        template_batch_size,
    )

    for angle_start in range(0, len(angles), angle_batch_size):
        chunk = angles[angle_start : angle_start + angle_batch_size]
        batch, valid = _build_rotated_batch(
            image, chunk, fill_value, canvas_shape, device
        )
        try:
            for template_start, scores in iterate_score_maps(
                batch,
                valid,
                template_tensor,
                minimum_fill_fraction,
                template_batch_size,
                mean_and_std,
                template_ffts=template_ffts,
            ):
                best.merge(scores, template_start, angle_start)
        finally:
            del batch, valid
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # A constant reference cell has no defined correlation, so it never gets a peak.
    return [
        best.find_peaks(index, angles, n_candidates, suppression_radius)
        if is_non_constant[index]
        else []
        for index in range(len(templates))
    ]


def find_best_matches(
    image: FloatArray2D,
    templates: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value: float,
    device: torch.device | None = None,
    template_batch_size: int | None = None,
    angle_batch_size: int | None = None,
) -> list[Match]:
    """
    Find the single best pose for every template over the full angle sweep.

    :func:`search_candidates` with ``n_candidates=1``. Useful both as a plain exhaustive matcher
    (the natural ground truth to check the coarse-to-fine search against on a small image) and as
    the pipeline's own fallback when the images already fit within the coarse-stage cap, where
    "coarse" and "fine" resolution coincide.

    :returns: One :class:`Match` per template. Templates with no viable candidate report the
        rejection sentinel ``Match(-1.0, 0, 0, default_angle)``.
    """
    candidates = search_candidates(
        image,
        templates,
        angles,
        minimum_fill_fraction,
        fill_value,
        n_candidates=1,
        template_batch_size=template_batch_size,
        angle_batch_size=angle_batch_size,
        device=device,
    )
    no_match = Match(-1.0, 0, 0, float(sort_by_absolute_angle(angles)[0]))
    return [found[0] if found else no_match for found in candidates]


def get_uniform_cell_shape(templates: list[np.ndarray]) -> tuple[int, int]:
    """Return the common ``(height, width)`` of *templates*, or raise if they disagree."""
    cell_shape = templates[0].shape
    if any(template.shape != cell_shape for template in templates):
        raise ValueError("All templates must have the same shape.")
    return cell_shape[0], cell_shape[1]


def sort_by_absolute_angle(angles: np.ndarray) -> np.ndarray:
    """Order an angle sweep by ``|angle|``, then by signed angle, so ties resolve predictably."""
    angles = np.asarray(angles, dtype=np.float64)
    return angles[np.lexsort((angles, np.abs(angles)))]


@dataclass
class _BestPoseGrid:
    """Running best score per ``(template, position)``, and which angle produced it."""

    best_score: torch.Tensor
    best_angle_index: torch.Tensor

    @classmethod
    def create_empty(
        cls, n_templates: int, out_shape: tuple[int, int], device: torch.device
    ) -> _BestPoseGrid:
        return cls(
            best_score=torch.full(
                (n_templates, *out_shape), REJECTED_SCORE, device=device
            ),
            # int16 is ample for any realistic sweep and this tensor dominates the per-chunk merge
            # traffic: at int64 it is 4x the bytes of best_score and is rewritten on every chunk.
            best_angle_index=torch.zeros(
                (n_templates, *out_shape), dtype=torch.int16, device=device
            ),
        )

    def merge(
        self, scores: torch.Tensor, template_start: int, angle_start: int
    ) -> None:
        """Fold one chunk's ``(n_angles, block, height, width)`` score volume into the running best."""
        chunk_best, chunk_best_within = scores.max(dim=0)
        dest = slice(template_start, template_start + scores.shape[1])
        # Masked in-place update rather than torch.where: after the first chunk only a few
        # positions improve, so this writes almost nothing, where torch.where rewrites both full
        # volumes every time. Basic slicing gives a view, so the assignment writes through.
        current = self.best_score[dest]
        better = chunk_best > current
        current[better] = chunk_best[better]
        self.best_angle_index[dest][better] = (
            angle_start + chunk_best_within[better]
        ).to(torch.int16)

    def find_peaks(
        self,
        index: int,
        angles: np.ndarray,
        n_candidates: int,
        suppression_radius: int,
    ) -> list[Match]:
        """Up to *n_candidates* well-separated maxima for one template, best first."""
        surface = self.best_score[index].clone()
        angle_index = self.best_angle_index[index]
        out_width = surface.shape[1]

        found: list[Match] = []
        for _ in range(n_candidates):
            y, x = divmod(int(surface.argmax()), out_width)
            score = float(surface[y, x])
            if score <= REJECTED_SCORE:
                break
            angle = float(angles[int(angle_index[y, x])])
            found.append(Match(clamp_score(score, index), x, y, angle))
            surface[
                max(0, y - suppression_radius) : y + suppression_radius + 1,
                max(0, x - suppression_radius) : x + suppression_radius + 1,
            ] = -np.inf
        return found


def _compute_common_canvas_shape(
    image_shape: tuple[int, ...], angles: np.ndarray
) -> tuple[int, int]:
    """
    Smallest canvas holding the image rotated to any angle in the sweep.

    Fixed for the whole sweep rather than sized per chunk: it keeps the transform size constant (so
    the FFT plan is built once and reused) and makes the result bit-for-bit independent of the
    chunk size, which matters when two candidates are exactly tied.
    """
    shapes = [
        compute_rotated_shape(image_shape[0], image_shape[1], float(angle))
        for angle in angles
    ]
    return max(shape[0] for shape in shapes), max(shape[1] for shape in shapes)


def _build_rotated_batch(
    image: FloatArray2D,
    angles: np.ndarray,
    fill_value: float,
    canvas_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate *image* at every angle in *angles* and stack the results onto *device*.

    Rotated canvases differ in size across angles, so each is written into the top-left corner of
    the common canvas. The slack is marked invalid, which the fill-fraction gate rejects, so it
    cannot produce a match.

    :returns: ``(batch, valid)``, both ``(n_angles, 1, *canvas_shape)`` float32 tensors.
    """
    shape = (len(angles), 1, *canvas_shape)
    batch = np.full(shape, fill_value, np.float32)
    valid = np.zeros(shape, np.float32)
    for index, angle in enumerate(angles):
        rotated = rotate_image(image, float(angle), fill_value=np.nan)
        is_data = ~np.isnan(rotated)
        rotated[~is_data] = fill_value
        height, width = rotated.shape
        batch[index, 0, :height, :width] = rotated
        valid[index, 0, :height, :width] = is_data
    return torch.from_numpy(batch).to(device), torch.from_numpy(valid).to(device)
