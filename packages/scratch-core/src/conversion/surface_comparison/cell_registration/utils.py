from __future__ import annotations

import logging
import math
from functools import lru_cache

import cv2
import numpy as np
import torch

from container_models.base import FloatArray2D
from conversion.surface_comparison.models import Cell, CellMetaData, GridCell
from conversion.surface_comparison.utils import convert_pixels_to_meters

logger = logging.getLogger(__name__)

SCORE_TOLERANCE = 0.01

#: Radices for which both pocketfft/MKL (CPU) and cuFFT (GPU) stay on their fast paths.
_FFT_RADICES = (2, 3, 5, 7)
#: Windows whose within-window sum of squares falls below ``_VARIANCE_EPS * n_pixels``
#: (on globally standardised data) are treated as constant and rejected.
_VARIANCE_EPS = 1e-8
_TINY = 1e-12
#: Marker written into score maps at positions that fail the fill or variance gate. It must sit
#: outside the valid Pearson range: -1.0 is a legitimate score (perfect anti-correlation), so using
#: it as a sentinel makes a genuinely inverted cell indistinguishable from a masked-out one.
REJECTED_SCORE = -2.0
#: Fraction of free memory the chunk planner is allowed to commit.
_MEMORY_FRACTION = 0.6
_DEFAULT_CPU_BUDGET = 4 * 1024**3
#: Upper bound on angles per chunk, and on templates correlated together.
#:
#: Small chunks win on both devices. A realistic sweep is already far past the point where extra
#: batching improves throughput, so larger chunks only inflate the working set: on CPU it stops
#: fitting in cache, and on GPU it wastes memory for no speed. Measured at 1500x1500, 150px cells,
#: 30 cells, 21 angles on a 12-core host with CUDA:
#:
#:   CUDA (64, 32) -> 1.55 s, 16.63 GB     CPU (16, 4) -> 13.63 s
#:   CUDA (16,  8) -> 1.48 s,  9.51 GB     CPU ( 8, 1) -> 14.07 s
#:   CUDA ( 8,  4) -> 1.21 s,  2.65 GB     CPU ( 2, 1) -> 13.75 s
#:   CUDA ( 4,  2) -> 1.25 s,  0.80 GB     CPU ( 1, 1) -> 12.47 s
#:
#: The CUDA setting below trades ~3% speed for ~3x less memory relative to (8, 4), which buys
#: headroom for larger scans, since peak allocation grows with image area. Re-tune with
#: ``tune_chunk_shape.py`` if the hardware or typical scan size changes.
_MAX_ANGLES_PER_CHUNK = {"cuda": 8, "cpu": 1}
_MAX_TEMPLATES_PER_CHUNK = {"cuda": 4, "cpu": 1}


def convert_grid_cell_to_cell(grid_cell: GridCell, pixel_size: float) -> Cell:
    """Convert an instance of `GridCell` to an instance of `Cell`."""
    cell = Cell(
        center_reference=convert_pixels_to_meters(
            values=grid_cell.center, pixel_size=pixel_size
        ),
        cell_size=convert_pixels_to_meters(
            values=(grid_cell.cell_data.shape[1], grid_cell.cell_data.shape[0]),
            pixel_size=pixel_size,
        ),
        fill_fraction_reference=grid_cell.fill_fraction,
        best_score=grid_cell.grid_search_params.score,
        angle_deg=grid_cell.grid_search_params.angle,
        center_comparison=convert_pixels_to_meters(
            values=(
                grid_cell.grid_search_params.center_x,
                grid_cell.grid_search_params.center_y,
            ),
            pixel_size=pixel_size,
        ),
        is_congruent=False,  # TODO: We shouldn't set this here?
        meta_data=CellMetaData(
            is_outlier=False, residual_angle_deg=0.0, position_error=(0, 0)
        ),  # TODO: We shouldn't set this here?
    )
    return cell


def pad_image_array(
    array: FloatArray2D, pad_width: int, pad_height: int, fill_value: float = np.nan
) -> FloatArray2D:
    """
    Pad a 2D array symmetrically with a constant fill value.

    Adds ``pad_height`` rows above and below and ``pad_width`` columns to the left and right of the input array.
    The original data is placed in the center of the output; the border is filled with ``fill_value``.

    :param array: Input 2D array of shape ``(height, width)``.
    :param pad_width: Number of columns to add on each side (left and right).
    :param pad_height: Number of rows to add on each side (top and bottom).
    :param fill_value: Constant value written into the padded border; defaults to NaN.
    :returns: Padded array of shape ``(height + 2 * pad_height, width + 2 * pad_width)``, same dtype as input.
    """
    height, width = array.shape
    new_shape = height + 2 * pad_height, width + 2 * pad_width
    output = np.full(shape=new_shape, fill_value=fill_value, dtype=array.dtype)
    output[pad_height : pad_height + height, pad_width : pad_width + width] = array
    return output


# --------------------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------------------


def rotated_shape(height: int, width: int, angle_deg: float) -> tuple[int, int]:
    """
    Output shape of :func:`rotate_image` without performing the rotation.

    Reproduces the sizing rule of ``skimage.transform.rotate(..., resize=True)``: the axis-aligned
    bounding box of the rotated corner points, rounded to the nearest integer.

    :param height: Height of the unrotated image in pixels.
    :param width: Width of the unrotated image in pixels.
    :param angle_deg: Rotation angle in degrees.
    :returns: ``(rotated_height, rotated_width)`` in pixels.
    """
    theta = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(theta)), abs(math.sin(theta))
    new_width = int(round((width - 1) * cos_a + (height - 1) * sin_a)) + 1
    new_height = int(round((width - 1) * sin_a + (height - 1) * cos_a)) + 1
    return new_height, new_width


def rotate_image(
    image: FloatArray2D,
    angle_deg: float,
    fill_value: float = np.nan,
) -> FloatArray2D:
    """
    Rotate *image* by *angle_deg* degrees, growing the canvas so no data is clipped.

    The rotation centre is ``((width - 1) / 2, (height - 1) / 2)`` and the centre of the input is
    mapped onto the centre of the output. Both conventions must stay in lock-step with
    :func:`unrotate_point`, which assumes exactly this mapping; see the round-trip test in
    ``test_cell_registration.py``.

    Nearest-neighbour interpolation is used so that no arithmetic is performed across the
    NaN boundary of the padded image.

    :param image: Input 2D array.
    :param angle_deg: Rotation angle in degrees (positive follows the same sense as the caller's sweep).
    :param fill_value: Value written outside the rotated source rectangle.
    :returns: Rotated float32 array of shape :func:`rotated_shape`.
    """
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    new_height, new_width = rotated_shape(height, width, angle_deg)

    matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    # ``getRotationMatrix2D`` maps ``center`` onto itself; shift it onto the new canvas centre.
    matrix[0, 2] += (new_width - 1) / 2.0 - center[0]
    matrix[1, 2] += (new_height - 1) / 2.0 - center[1]

    return np.asarray(
        cv2.warpAffine(
            image.astype(np.float32),
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(fill_value),
        ),
        dtype=np.float32,
    )


def rotated_crop(
    image: FloatArray2D,
    angle_deg: float,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    fill_value: float = np.nan,
) -> FloatArray2D:
    """
    Produce a crop of the rotated canvas without rotating the whole image.

    ``warpAffine`` maps each *output* pixel back into the source, so folding the crop offset into
    the translation yields exactly the same pixels as rotating the full image and slicing it.

    :param image: Source image, already float32.
    :param angle_deg: Rotation angle in degrees.
    :param left: Left edge of the desired crop, in rotated-canvas coordinates.
    :param top: Top edge of the desired crop, in rotated-canvas coordinates.
    :param crop_width: Crop width in pixels.
    :param crop_height: Crop height in pixels.
    :param fill_value: Value written outside the rotated source rectangle.
    :returns: Float32 array of shape ``(crop_height, crop_width)``.
    """
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotated_height, rotated_width = rotated_shape(height, width, angle_deg)

    matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    matrix[0, 2] += (rotated_width - 1) / 2.0 - center[0] - left
    matrix[1, 2] += (rotated_height - 1) / 2.0 - center[1] - top

    return cv2.warpAffine(
        image,
        matrix,
        (crop_width, crop_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )


def canvas_to_image(
    x: float,
    y: float,
    cell_shape: tuple[int, int],
    image_shape: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    """
    Map a matched window on a rotated canvas back to a cell centre in the unrotated image.

    Inverse of :func:`image_to_canvas`. Both assume the rotation conventions of
    :func:`rotate_image`; a round-trip test guards that they stay in step.

    :param x: Window left edge on the rotated canvas.
    :param y: Window top edge on the rotated canvas.
    :param cell_shape: ``(cell_height, cell_width)``.
    :param image_shape: ``(height, width)`` of the unrotated image.
    :param angle_deg: Rotation angle in degrees.
    :returns: The window's centre in unrotated image coordinates.
    """
    cell_height, cell_width = cell_shape
    height, width = image_shape
    rotated_height, rotated_width = rotated_shape(height, width, angle_deg)

    dx = x + cell_width / 2 - (rotated_width - 1) / 2
    dy = y + cell_height / 2 - (rotated_height - 1) / 2
    cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return (
        (width - 1) / 2 + cos_a * dx + sin_a * dy,
        (height - 1) / 2 - sin_a * dx + cos_a * dy,
    )


def image_to_canvas(
    center_x: float,
    center_y: float,
    cell_shape: tuple[int, int],
    image_shape: tuple[int, int],
    angle_deg: float,
) -> tuple[float, float]:
    """
    Map a cell centre in the unrotated image to a window's top-left corner on the rotated canvas.

    Inverse of :func:`canvas_to_image`.

    :param center_x: Cell centre x in unrotated image coordinates.
    :param center_y: Cell centre y in unrotated image coordinates.
    :param cell_shape: ``(cell_height, cell_width)``.
    :param image_shape: ``(height, width)`` of the unrotated image.
    :param angle_deg: Rotation angle in degrees.
    :returns: The window's top-left corner on the rotated canvas.
    """
    cell_height, cell_width = cell_shape
    height, width = image_shape
    rotated_height, rotated_width = rotated_shape(height, width, angle_deg)

    dx = center_x - (width - 1) / 2
    dy = center_y - (height - 1) / 2
    cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return (
        cos_a * dx - sin_a * dy + (rotated_width - 1) / 2 - cell_width / 2,
        sin_a * dx + cos_a * dy + (rotated_height - 1) / 2 - cell_height / 2,
    )


# --------------------------------------------------------------------------------------
# Device-agnostic building blocks
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def next_fast_len(target: int) -> int:
    """
    Smallest integer ``>= target`` that factors entirely into :data:`_FFT_RADICES`.

    Transform lengths with a large prime factor fall off the fast path in both MKL/pocketfft and
    cuFFT, which is routinely a 2-5x penalty on the sizes used here.
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


def box_sum(
    values: torch.Tensor,
    window_height: int,
    window_width: int,
    accumulate_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sum over every window with its **top-left corner** at ``[y, x]``, via a summed-area table.

    This replaces the FFT-based sliding sums used previously. It is O(1) per output pixel instead
    of O(log n), works identically on CPU and CUDA, and matches the indexing convention of
    ``cv2.matchTemplate`` (which computes its normalisation the same way internally).

    The accumulation runs in float64 by default: the running total over a ~3000x3000 image reaches
    ~1e7, and differencing two such float32 values to recover a window sum of ~1e4 loses roughly
    three significant digits. Passing ``torch.float32`` halves the transient memory at that cost.

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


def _correlate_valid(
    image_fft: torch.Tensor,
    templates: torch.Tensor,
    fft_height: int,
    fft_width: int,
    out_height: int,
    out_width: int,
) -> torch.Tensor:
    """
    Cross-correlate a pre-transformed image batch with a block of templates ("valid" mode).

    ``image_fft`` is supplied already transformed so that it can be computed **once** per angle
    chunk and reused across every template; recomputing it inside the template loop was the
    single largest cost in the previous implementation.

    :param image_fft: Half-spectrum of the image batch, shape ``(n_angles, 1, fft_height, fft_width // 2 + 1)``.
    :param templates: Template block of shape ``(n_templates, 1, cell_height, cell_width)``.
    :param fft_height: Transform height (see :func:`next_fast_len`).
    :param fft_width: Transform width.
    :param out_height: Number of valid output rows.
    :param out_width: Number of valid output columns.
    :returns: Contiguous float32 tensor ``(n_angles, n_templates, out_height, out_width)``.
    """
    template_fft = torch.fft.rfft2(templates.transpose(0, 1), s=(fft_height, fft_width))
    correlation = torch.fft.irfft2(
        image_fft * template_fft.conj(), s=(fft_height, fft_width)
    )
    return correlation[..., :out_height, :out_width].contiguous()


# --------------------------------------------------------------------------------------
# Batched matching
# --------------------------------------------------------------------------------------


def _prepare_rotated_batch(
    image: FloatArray2D,
    angles: np.ndarray,
    fill_value: float,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate *image* at every angle in *angles* and stack the results into a fixed-size batch.

    Rotated canvases differ in size across angles, so each is written into the top-left corner of a
    common canvas. The slack is marked invalid, which the fill-fraction gate rejects, so it cannot
    produce a match.

    *canvas_shape* is deliberately fixed for the whole sweep rather than sized per chunk. It keeps
    the transform size constant (so the FFT plan is built once and reused) and makes the result
    bit-for-bit independent of the chunk size, which matters when two candidates are exactly tied.

    :param image: Padded comparison image, NaN outside the original data.
    :param angles: Angles for this chunk, in degrees.
    :param fill_value: Value substituted for NaN.
    :param canvas_shape: ``(height, width)`` of the common canvas; must fit every rotated image.
    :returns: ``(batch, valid_batch)``, both shaped ``(n_angles, 1, *canvas_shape)`` float32.
    """
    canvas_height, canvas_width = canvas_shape
    batch = np.full((len(angles), 1, canvas_height, canvas_width), fill_value, np.float32)
    valid_batch = np.zeros((len(angles), 1, canvas_height, canvas_width), np.float32)
    for index, angle in enumerate(angles):
        rotated = rotate_image(image, float(angle), fill_value=np.nan)
        valid = ~np.isnan(rotated)
        rotated[~valid] = fill_value
        height, width = rotated.shape
        batch[index, 0, :height, :width] = rotated
        valid_batch[index, 0, :height, :width] = valid
    return batch, valid_batch


def _prepare_templates(
    templates: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Stack templates onto *device*, centred and scaled to unit norm.

    Pearson correlation is invariant to a positive affine rescaling of either operand, so the
    template can be normalised independently of the image. Doing so keeps the float32 FFT
    well-conditioned and removes the template norm from the score denominator entirely.

    :returns: ``(tensor of shape (n_templates, 1, cell_height, cell_width), is_non_constant mask)``.
    """
    stacked = np.stack([template.astype(np.float32) for template in templates])[:, None]
    tensor = torch.from_numpy(stacked).to(device).clone()
    tensor -= tensor.mean(dim=(2, 3), keepdim=True)
    norms = tensor.flatten(1).norm(dim=1)
    is_non_constant = (norms > _TINY).cpu().numpy()
    tensor /= norms.clamp_min(_TINY).view(-1, 1, 1, 1)
    return tensor, is_non_constant


def _available_bytes(device: torch.device) -> int:
    """Memory the chunk planner may commit on *device*."""
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes * _MEMORY_FRACTION)
    try:
        import psutil

        return int(psutil.virtual_memory().available * _MEMORY_FRACTION)
    except Exception:  # pragma: no cover - psutil is optional
        logger.debug("psutil unavailable; falling back to a fixed CPU memory budget.")
        return _DEFAULT_CPU_BUDGET


def plan_chunks(
    device: torch.device,
    n_angles: int,
    n_templates: int,
    fft_height: int,
    fft_width: int,
    memory_budget: int | None = None,
) -> tuple[int, int]:
    """
    Choose ``(angles_per_chunk, templates_per_chunk)`` that fit a memory budget.

    This is the only place where CPU and CUDA are treated differently, and only as a tuning knob:
    the algorithm below is identical on both. The estimate is deliberately coarse and conservative;
    pass *memory_budget* explicitly to pin it in benchmarks.

    :param device: Target device.
    :param n_angles: Total number of angles in the sweep.
    :param n_templates: Total number of grid cells.
    :param fft_height: Transform height for the largest rotated canvas.
    :param fft_width: Transform width for the largest rotated canvas.
    :param memory_budget: Byte budget; defaults to a fraction of free device memory.
    :returns: ``(angles_per_chunk, templates_per_chunk)``, both at least 1.
    """
    if memory_budget is None:
        memory_budget = _available_bytes(device)

    complex_bytes = fft_height * (fft_width // 2 + 1) * 8
    real_bytes = fft_height * fft_width * 4
    # Live per angle: image half-spectrum, rotated batch, validity mask, two box-sum outputs,
    # plus (transiently) one float64 summed-area table.
    per_angle = complex_bytes + 4 * real_bytes + 2 * real_bytes
    # Live per (angle, template) pair: the complex product, the inverse transform, and its
    # contiguous "valid" slice.
    per_pair = complex_bytes + 2 * real_bytes

    angles_per_chunk = max(1, min(n_angles, _MAX_ANGLES_PER_CHUNK.get(device.type, 8)))
    while (
        angles_per_chunk > 1
        and angles_per_chunk * (per_angle + per_pair) > memory_budget
    ):
        angles_per_chunk //= 2

    remaining = memory_budget - angles_per_chunk * per_angle
    templates_per_chunk = (
        int(remaining // (angles_per_chunk * per_pair)) if remaining > 0 else 1
    )
    templates_per_chunk = max(
        1,
        min(
            n_templates,
            templates_per_chunk,
            _MAX_TEMPLATES_PER_CHUNK.get(device.type, 1),
        ),
    )
    return angles_per_chunk, templates_per_chunk


def iter_score_maps(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: torch.Tensor,
    minimum_fill_fraction: float,
    templates_per_chunk: int,
    standardisation: tuple[float, float],
):
    """
    Yield normalised cross-correlation maps for one batch of rotated images.

    Computes ``r = sum(W * T') / (sqrt(sum((W - mean(W))^2)) * ||T'||)`` where ``T'`` is the centred,
    unit-norm template. Because ``T'`` sums to zero, the local mean of ``W`` cancels out of the
    numerator, so only the local sum and sum of squares are needed - both from summed-area tables
    rather than extra transforms. Rejected positions are set to :data:`REJECTED_SCORE`.

    Both the exhaustive and the coarse-to-fine search consume this, so the scoring math has exactly
    one implementation.

    :param batch: Rotated image batch ``(n_angles, 1, height, width)``.
    :param valid: Validity mask ``(n_angles, 1, height, width)``, 1.0 where the pixel holds real data.
    :param templates: Centred unit-norm templates ``(n_templates, 1, cell_height, cell_width)``.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param templates_per_chunk: Templates correlated per iteration.
    :param standardisation: ``(mean, standard_deviation)`` of the whole comparison image. These are
        deliberately global rather than per-chunk: correlation is invariant to the rescaling, but
        the float32 rounding it induces is not, and per-chunk statistics would make the result
        depend on the chunk size.
    :yields: ``(template_offset, scores)`` where scores has shape
        ``(n_angles, block_size, out_height, out_width)``.
    """
    n_angles, _, height, width = batch.shape
    n_templates, _, cell_height, cell_width = templates.shape
    n_pixels = cell_height * cell_width
    out_height, out_width = height - cell_height + 1, width - cell_width + 1

    fill_fraction = box_sum(valid, cell_height, cell_width) / n_pixels
    position_ok = fill_fraction >= minimum_fill_fraction
    del fill_fraction

    # Standardise the batch. Pearson correlation is invariant to this, but it keeps the float32
    # transforms well away from the precision cliff on large canvases. This is also what makes the
    # rewrite measurably more accurate than ``cv2.matchTemplate`` on data with a large DC offset.
    mean, standard_deviation = standardisation
    batch = (batch - mean) / max(standard_deviation, _TINY)

    local_sum = box_sum(batch, cell_height, cell_width)
    local_sum_of_squares = box_sum(batch * batch, cell_height, cell_width)
    local_variation = local_sum_of_squares - local_sum.square() / n_pixels
    del local_sum, local_sum_of_squares

    position_ok &= local_variation > _VARIANCE_EPS * n_pixels
    denominator = local_variation.clamp_min(0.0).sqrt().clamp_min(_TINY)
    del local_variation
    rejected = ~position_ok
    del position_ok

    fft_height = next_fast_len(height + cell_height - 1)
    fft_width = next_fast_len(width + cell_width - 1)
    image_fft = torch.fft.rfft2(batch, s=(fft_height, fft_width))
    del batch

    for start in range(0, n_templates, templates_per_chunk):
        block = templates[start : start + templates_per_chunk]
        scores = _correlate_valid(
            image_fft, block, fft_height, fft_width, out_height, out_width
        )
        scores.div_(denominator).masked_fill_(rejected, REJECTED_SCORE)
        yield start, scores


def paired_score_maps(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: torch.Tensor,
    minimum_fill_fraction: float,
    standardisation: tuple[float, float],
) -> torch.Tensor:
    """
    Correlate each image in *batch* against its own template, pairwise rather than all-against-all.

    :func:`iter_score_maps` computes the cross product of images and templates, which is what an
    exhaustive sweep wants. Local refinement wants the opposite: many small crops, each with one
    specific template. Pairing them keeps a single large batch instead of thousands of tiny calls,
    where per-call overhead would otherwise dominate.

    :param batch: Image crops ``(n, 1, height, width)``.
    :param valid: Validity masks ``(n, 1, height, width)``, 1.0 where the pixel holds real data.
    :param templates: Centred unit-norm templates ``(n, 1, cell_height, cell_width)``, aligned with *batch*.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param standardisation: Global ``(mean, standard_deviation)`` of the comparison image.
    :returns: Scores ``(n, 1, out_height, out_width)``, rejected positions set to
        :data:`REJECTED_SCORE`.
    """
    n, _, height, width = batch.shape
    cell_height, cell_width = templates.shape[2], templates.shape[3]
    n_pixels = cell_height * cell_width
    out_height, out_width = height - cell_height + 1, width - cell_width + 1

    fill_fraction = box_sum(valid, cell_height, cell_width) / n_pixels
    position_ok = fill_fraction >= minimum_fill_fraction
    del fill_fraction

    mean, standard_deviation = standardisation
    batch = (batch - mean) / max(standard_deviation, _TINY)

    local_sum = box_sum(batch, cell_height, cell_width)
    local_variation = (
        box_sum(batch * batch, cell_height, cell_width) - local_sum.square() / n_pixels
    )
    del local_sum

    position_ok &= local_variation > _VARIANCE_EPS * n_pixels
    denominator = local_variation.clamp_min(0.0).sqrt().clamp_min(_TINY)
    del local_variation

    fft_height = next_fast_len(height + cell_height - 1)
    fft_width = next_fast_len(width + cell_width - 1)
    image_fft = torch.fft.rfft2(batch, s=(fft_height, fft_width))
    template_fft = torch.fft.rfft2(templates, s=(fft_height, fft_width))
    scores = torch.fft.irfft2(image_fft * template_fft.conj(), s=(fft_height, fft_width))
    del image_fft, template_fft

    scores = scores[..., :out_height, :out_width].contiguous()
    scores.div_(denominator).masked_fill_(~position_ok, REJECTED_SCORE)
    return scores


def _match_chunk(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: torch.Tensor,
    minimum_fill_fraction: float,
    templates_per_chunk: int,
    standardisation: tuple[float, float],
) -> list[tuple[float, int, int, int]]:
    """
    Best ``(score, x, y, angle_index_within_chunk)`` per template for one angle chunk.

    :param batch: Rotated image batch ``(n_angles, 1, height, width)``.
    :param valid: Validity mask, 1.0 where the pixel holds real data.
    :param templates: Centred unit-norm templates.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param templates_per_chunk: Templates correlated per iteration.
    :param standardisation: Global ``(mean, standard_deviation)`` of the comparison image.
    :returns: Per template, ``(score, x, y, angle_index_within_chunk)``.
    """
    device = batch.device
    n_angles = batch.shape[0]
    out_width = batch.shape[3] - templates.shape[3] + 1

    results: list[tuple[float, int, int, int]] = []
    for _, scores in iter_score_maps(
        batch,
        valid,
        templates,
        minimum_fill_fraction,
        templates_per_chunk,
        standardisation,
    ):
        n_block = scores.shape[1]
        best_per_angle, position_per_angle = scores.reshape(n_angles, n_block, -1).max(
            dim=2
        )
        del scores
        best_value, best_angle = best_per_angle.max(dim=0)
        best_position = position_per_angle[
            best_angle, torch.arange(n_block, device=device)
        ]
        for value, angle_index, position in zip(
            best_value.tolist(), best_angle.tolist(), best_position.tolist()
        ):
            results.append(
                (
                    float(value),
                    int(position % out_width),
                    int(position // out_width),
                    int(angle_index),
                )
            )
    return results


def batched_match(
    image: FloatArray2D,
    templates: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value: float,
    device: torch.device | None = None,
    memory_budget: int | None = None,
) -> list[tuple[float, int, int, float]]:
    """
    Find the best (score, position, angle) for every template over the full angle sweep.

    One code path serves CPU and CUDA; only the chunk sizes differ, via :func:`plan_chunks`.

    Angles are processed in order of increasing ``|angle|``. This makes the rotated canvases within
    a chunk close in size (so less of the batch is padding), and it makes the tie-break meaningful:
    ``torch.max`` returns the first maximal index, so exact ties resolve to the smallest ``|angle|``,
    then the smallest ``y``, then the smallest ``x``.

    :param image: Padded comparison image, NaN outside the original data.
    :param templates: Reference cell data, all the same shape and free of NaN.
    :param angles: Angle sweep in degrees.
    :param minimum_fill_fraction: Reject positions whose window is filled below this fraction.
    :param fill_value: Value substituted for NaN in the comparison image.
    :param device: Torch device; defaults to CUDA when available.
    :param memory_budget: Optional explicit byte budget for chunk planning.
    :returns: Per template, ``(score, x, y, angle_deg)`` with ``x``/``y`` in rotated-canvas pixels.
    """
    if not templates:
        return []
    cell_shape = templates[0].shape
    if any(template.shape != cell_shape for template in templates):
        raise ValueError("All templates must have the same shape.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    angles = np.asarray(angles, dtype=np.float64)
    sorted_angles = angles[np.lexsort((angles, np.abs(angles)))]

    cell_height, cell_width = cell_shape
    height, width = image.shape
    shapes = [rotated_shape(height, width, float(angle)) for angle in sorted_angles]
    canvas_shape = (max(shape[0] for shape in shapes), max(shape[1] for shape in shapes))
    fft_height = next_fast_len(canvas_shape[0] + cell_height - 1)
    fft_width = next_fast_len(canvas_shape[1] + cell_width - 1)

    angles_per_chunk, templates_per_chunk = plan_chunks(
        device, len(sorted_angles), len(templates), fft_height, fft_width, memory_budget
    )
    logger.debug(
        "Matching %d templates over %d angles on %s (chunks: %d angles x %d templates).",
        len(templates),
        len(sorted_angles),
        device,
        angles_per_chunk,
        templates_per_chunk,
    )

    template_tensor, is_non_constant = _prepare_templates(templates, device)
    standardisation = (float(np.nanmean(image)), float(np.nanstd(image)))

    best: list[tuple[float, int, int, float]] = [
        (-np.inf, 0, 0, float(sorted_angles[0])) for _ in templates
    ]
    for start in range(0, len(sorted_angles), angles_per_chunk):
        chunk_angles = sorted_angles[start : start + angles_per_chunk]
        batch, valid = _prepare_rotated_batch(
            image, chunk_angles, fill_value, canvas_shape
        )
        batch_tensor = torch.from_numpy(batch).to(device)
        valid_tensor = torch.from_numpy(valid).to(device)
        try:
            chunk_results = _match_chunk(
                batch_tensor,
                valid_tensor,
                template_tensor,
                minimum_fill_fraction,
                templates_per_chunk,
                standardisation,
            )
        finally:
            del batch_tensor, valid_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for index, (score, x, y, angle_index) in enumerate(chunk_results):
            if score > best[index][0]:
                best[index] = (score, x, y, float(chunk_angles[angle_index]))

    for index, (score, x, y, angle) in enumerate(best):
        if not is_non_constant[index]:
            # A constant reference cell has no defined correlation.
            best[index] = (-1.0, 0, 0, float(sorted_angles[0]))
            continue
        if score > 1.0 + SCORE_TOLERANCE:
            logger.warning(
                "NCC score %.4f exceeds the valid range [-1, 1] for cell %d; clamping.",
                score,
                index,
            )
        # A cell whose every position was rejected still reports the floor of the valid range.
        best[index] = (min(max(score, -1.0), 1.0), x, y, angle)
    return best