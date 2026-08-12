"""
Normalized cross-correlation utilities.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
from loguru import logger

from container_models.base import FloatArray2D

#: A reported score may overshoot 1.0 by this much through float32 rounding before it is worth a
#: warning; anything above 1.0 is clamped either way.
SCORE_TOLERANCE = 0.01

#: Marker written into score maps at positions failing the fill or variance gate. It must sit
#: outside the valid Pearson range: -1.0 is a legitimate score (perfect anti-correlation), so using
#: it would make a genuinely inverted cell indistinguishable from a masked-out one.
REJECTED_SCORE = -2.0

#: Radices for which both pocketfft/MKL (CPU) and cuFFT (GPU) stay on their fast paths.
#: 7 is excluded as it is slower than 2, 3, or 5.
_FFT_RADICES = (2, 3, 5)
#: Windows whose within-window sum of squares falls below ``_VARIANCE_EPS * n_pixels``
#: (on globally standardized data) are treated as constant and rejected.
_VARIANCE_EPS = 1e-8
_TINY = 1e-12


def compute_mean_and_std(image: FloatArray2D) -> tuple[float, float]:
    """Global mean and standard deviation of an image."""
    return float(np.nanmean(image)), float(np.nanstd(image))


def clamp_score(score: float, index: int) -> float:
    """Clamp a score into the valid Pearson range, warning if it overshot by more than tolerance."""
    if score > 1.0 + SCORE_TOLERANCE:
        logger.warning(
            "NCC score {:.4f} exceeds the valid range [-1, 1] for cell {}; clamping.",
            score,
            index,
        )
    return min(max(score, -1.0), 1.0)


@cache
def find_next_fast_length(target: int) -> int:
    """
    Smallest integer ``>= target`` that factors entirely into _FFT_RADICES.

    Transform lengths with a large prime factor are significantly slower in MKL/pocketfft and cuFFT.
    """
    if target <= 2:
        return max(int(target), 1)
    candidate = int(target)
    while True:
        remainder = candidate
        for radix in _FFT_RADICES:
            while remainder % radix == 0:
                remainder //= radix
        if remainder == 1:
            return candidate
        candidate += 1


def sum_over_windows(
    values: torch.Tensor,
    window_height: int,
    window_width: int,
    accumulate_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sum over every window with its **top-left corner** at ``[y, x]``, via a summed-area table.

    O(1) per output pixel, matching the indexing convention of ``cv2.matchTemplate``.
    The accumulation runs in float64 by default to avoid precision loss.

    :param values: Tensor of shape ``(batch, 1, height, width)``.
    :param window_height: Window height in pixels.
    :param window_width: Window width in pixels.
    :param accumulate_dtype: dtype of the summed-area table.
    :returns: Float32 tensor of shape ``(batch, 1, height - window_height + 1, width - window_width + 1)``.
    """
    integral = torch.nn.functional.pad(values.to(accumulate_dtype), (1, 0, 1, 0))
    integral = integral.cumsum_(dim=-1).cumsum_(dim=-2)
    result = (
        integral[..., window_height:, window_width:]
        - integral[..., :-window_height, window_width:]
        - integral[..., window_height:, :-window_width]
        + integral[..., :-window_height, :-window_width]
    )
    return result.to(torch.float32)


def prepare_templates(
    templates: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Stack templates onto *device*, centered and scaled to unit norm.

    Template normalization is done independently of the image to keep the FFT well-conditioned.

    :returns: ``(tensor of shape (n_templates, 1, cell_height, cell_width), is_non_constant mask)``.
    """
    stacked = np.stack([template.astype(np.float32) for template in templates])[:, None]
    tensor = torch.from_numpy(stacked).to(device).clone()
    tensor -= tensor.mean(dim=(2, 3), keepdim=True)
    norms = tensor.flatten(1).norm(dim=1)
    is_non_constant = (norms > _TINY).cpu().numpy()
    tensor /= norms.clamp_min(_TINY).view(-1, 1, 1, 1)
    return tensor, is_non_constant


def precompute_template_ffts(
    templates: torch.Tensor, fft_shape: tuple[int, int], templates_per_chunk: int
) -> list[tuple[int, torch.Tensor]]:
    """
    Transform every template block once, for reuse across all angle chunks.

    :returns: ``(template_offset, spectrum)`` per block, the spectrum shaped ``(1, block, ...)`` so
        it broadcasts against an image batch of shape ``(n_angles, 1, ...)``.
    """
    return [
        (
            start,
            torch.fft.rfft2(
                templates[start : start + templates_per_chunk].transpose(0, 1),
                s=fft_shape,
            ),
        )
        for start in range(0, templates.shape[0], templates_per_chunk)
    ]


@dataclass(frozen=True)
class CorrelationBasis:
    """
    Everything about one image batch that does not depend on which template it is scored against.

    Built once per batch by build_correlation_basis, then combined with any number of
    template spectra by compute_scores.
    """

    image_fft: torch.Tensor
    #: Per-position ``sqrt`` of the within-window variation, i.e. the score denominator.
    denominator: torch.Tensor
    #: Per-position mask of windows failing the fill-fraction or variance gate.
    rejected: torch.Tensor
    fft_shape: tuple[int, int]
    out_shape: tuple[int, int]

    def compute_scores(self, template_fft: torch.Tensor) -> torch.Tensor:
        """
        Score the batch against one block of pre-transformed templates.

        *template_fft* must broadcast against image_fft; rejected positions come back as
        REJECTED_SCORE.
        """
        out_height, out_width = self.out_shape
        correlation = torch.fft.irfft2(
            self.image_fft * template_fft.conj(), s=self.fft_shape
        )
        scores = correlation[..., :out_height, :out_width].contiguous()
        return scores.div_(self.denominator).masked_fill_(self.rejected, REJECTED_SCORE)


def build_correlation_basis(
    batch: torch.Tensor,
    valid: torch.Tensor,
    cell_shape: tuple[int, int],
    minimum_fill_fraction: float,
    mean_and_std: tuple[float, float],
) -> CorrelationBasis:
    """
    Gate, standardize and transform one image batch.

    :param batch: Image batch ``(n, 1, height, width)``.
    :param valid: Validity mask of the same shape, 1.0 where the pixel holds real data.
    :param cell_shape: ``(cell_height, cell_width)`` of the templates to be scored.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param mean_and_std: Statistics of the whole comparison image.
    """
    cell_height, cell_width = cell_shape
    n_pixels = cell_height * cell_width
    height, width = batch.shape[-2:]

    fill_fraction = sum_over_windows(valid, cell_height, cell_width) / n_pixels
    position_ok = fill_fraction >= minimum_fill_fraction
    del fill_fraction

    # Pearson correlation is invariant to standardization, but it keeps the float32 transforms
    # well away from the precision cliff on large canvases.
    mean, standard_deviation = mean_and_std
    batch = (batch - mean) / max(standard_deviation, _TINY)

    local_sum = sum_over_windows(batch, cell_height, cell_width)
    variation = (
        sum_over_windows(batch * batch, cell_height, cell_width)
        - local_sum.square() / n_pixels
    )
    del local_sum

    position_ok &= variation > _VARIANCE_EPS * n_pixels
    denominator = variation.clamp_min(0.0).sqrt().clamp_min(_TINY)
    del variation

    fft_shape = (
        find_next_fast_length(height + cell_height - 1),
        find_next_fast_length(width + cell_width - 1),
    )
    return CorrelationBasis(
        image_fft=torch.fft.rfft2(batch, s=fft_shape),
        denominator=denominator,
        rejected=~position_ok,
        fft_shape=fft_shape,
        out_shape=(height - cell_height + 1, width - cell_width + 1),
    )


def iterate_score_maps(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: torch.Tensor,
    minimum_fill_fraction: float,
    templates_per_chunk: int,
    mean_and_std: tuple[float, float],
    template_ffts: list[tuple[int, torch.Tensor]] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """
    Score every image in *batch* against every template: the cross product an exhaustive sweep wants.

    :param batch: Rotated image batch ``(n_angles, 1, height, width)``.
    :param valid: Validity mask of the same shape.
    :param templates: Centered unit-norm templates ``(n_templates, 1, cell_height, cell_width)``.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param templates_per_chunk: Templates correlated per iteration.
    :param mean_and_std: Global statistics of the comparison image.
    :param template_ffts: Pre-transformed templates from precompute_template_ffts, when the
        transform size is fixed across batches and they can be computed once for the whole sweep.
    :yields: ``(template_offset, scores)``, scores shaped ``(n_angles, block, out_height, out_width)``.
    """
    cell_shape = (templates.shape[2], templates.shape[3])
    basis = build_correlation_basis(
        batch, valid, cell_shape, minimum_fill_fraction, mean_and_std
    )
    if template_ffts is None:
        template_ffts = precompute_template_ffts(
            templates, basis.fft_shape, templates_per_chunk
        )
    for start, template_fft in template_ffts:
        yield start, basis.compute_scores(template_fft)


def compute_paired_score_maps(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: torch.Tensor,
    minimum_fill_fraction: float,
    mean_and_std: tuple[float, float],
) -> torch.Tensor:
    """
    Score each image in *batch* against its own template, 1:1 rather than all-against-all.

    Pairing them keeps a single large batch instead of thousands of tiny calls, where per-call
    overhead would otherwise dominate.

    :param batch: Image crops ``(n, 1, height, width)``.
    :param valid: Validity mask of the same shape.
    :param templates: Centered unit-norm templates ``(n, 1, cell_height, cell_width)``, aligned with *batch*.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param mean_and_std: Global statistics of the comparison image.
    :returns: Scores ``(n, 1, out_height, out_width)``.
    """
    cell_shape = (templates.shape[2], templates.shape[3])
    basis = build_correlation_basis(
        batch, valid, cell_shape, minimum_fill_fraction, mean_and_std
    )
    return basis.compute_scores(torch.fft.rfft2(templates, s=basis.fft_shape))
