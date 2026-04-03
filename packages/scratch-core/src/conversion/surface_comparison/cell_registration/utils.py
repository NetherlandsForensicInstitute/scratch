from __future__ import annotations

import cv2
import torch

from container_models.base import FloatArray2D
from conversion.surface_comparison.models import Cell, GridCell, CellMetaData

import numpy as np

from conversion.surface_comparison.utils import convert_pixels_to_meters


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


def _prepare_rotated_batch(
    image: FloatArray2D,
    angles: np.ndarray,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Rotate *image* at every angle and stack into padded arrays.

    :returns: (batch, valid_batch, rotated_shapes) where batch and
        valid_batch have shape ``(n_angles, 1, max_h, max_w)``.
    """
    rotated_list: list[np.ndarray] = []
    valid_list: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []

    for angle in angles:
        rotated = _rotate_image(image, float(angle), fill_value=np.nan)
        valid = ~np.isnan(rotated)
        rotated_clean = rotated.copy()
        rotated_clean[~valid] = fill_value
        rotated_list.append(rotated_clean)
        valid_list.append(valid.astype(np.float32))
        shapes.append(rotated.shape)

    max_h = max(r.shape[0] for r in rotated_list)
    max_w = max(r.shape[1] for r in rotated_list)

    batch = np.full((len(angles), 1, max_h, max_w), fill_value, dtype=np.float32)
    valid_batch = np.zeros((len(angles), 1, max_h, max_w), dtype=np.float32)
    for i, (r, v) in enumerate(zip(rotated_list, valid_list)):
        h, w = r.shape
        batch[i, 0, :h, :w] = r
        valid_batch[i, 0, :h, :w] = v

    return batch, valid_batch, shapes


def _fft_match(image_batch: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """FFT-based cross-correlation (``valid`` mode) for a batch of images."""
    _, _, h_img, w_img = image_batch.shape
    h_t, w_t = template.shape[2], template.shape[3]

    pad_h = h_img + h_t - 1
    pad_w = w_img + w_t - 1

    img_fft = torch.fft.rfft2(image_batch, s=(pad_h, pad_w))
    templ_fft = torch.fft.rfft2(template, s=(pad_h, pad_w))
    result = torch.fft.irfft2(img_fft * templ_fft.conj(), s=(pad_h, pad_w))

    out_h = h_img - h_t + 1
    out_w = w_img - w_t + 1
    return result[:, :, :out_h, :out_w]


def _ncc(
    batch: torch.Tensor,
    valid: torch.Tensor,
    templates: list[np.ndarray],
    minimum_fill_fraction: float,
    device: torch.device,
) -> list[tuple[float, int, int, int]]:
    """
    Normalised cross-correlation of every template against a batch of
    rotated images on GPU.

    :returns: Per-template list of ``(score, x, y, angle_index)``.
    """
    cell_h, cell_w = templates[0].shape
    n_pixels = cell_h * cell_w

    ones_kern = torch.ones(1, 1, cell_h, cell_w, device=device)

    fill_map = _fft_match(valid, ones_kern / n_pixels)
    fill_mask = fill_map >= minimum_fill_fraction

    # Normalize batch to avoid precision issues with small values
    batch_mean = batch.mean()
    batch_std = batch.std()
    if batch_std > 0:
        batch_normed = (batch - batch_mean) / batch_std
    else:
        batch_normed = batch - batch_mean

    local_sum = _fft_match(batch_normed, ones_kern)
    local_sq_sum = _fft_match(batch_normed**2, ones_kern)
    local_mean = local_sum / n_pixels
    local_var = local_sq_sum / n_pixels - local_mean**2

    n_angles = batch.shape[0]
    neg_one = torch.tensor(-1.0, device=device)
    results = []

    for template in templates:
        templ_t = torch.from_numpy(template.astype(np.float32)).to(device)
        # Normalize template with same stats
        if batch_std > 0:
            templ_normed = (templ_t - batch_mean) / batch_std
        else:
            templ_normed = templ_t - batch_mean
        templ_kern = templ_normed.unsqueeze(0).unsqueeze(0)
        t_mean = templ_normed.mean()
        t_std = templ_normed.std()

        cross = _fft_match(batch_normed, templ_kern) / n_pixels
        numerator = cross - local_mean * t_mean
        denominator = torch.sqrt(torch.clamp(local_var, min=0)) * t_std

        score_maps = torch.where(denominator > 1e-7, numerator / denominator, neg_one)
        score_maps = torch.where(
            fill_mask[:, :, : score_maps.shape[2], : score_maps.shape[3]],
            score_maps,
            neg_one,
        )

        flat = score_maps.view(n_angles, -1)
        max_per_angle, pos_per_angle = flat.max(dim=1)
        best_angle_idx = int(max_per_angle.argmax())
        best_score = float(max_per_angle[best_angle_idx])

        score_w = score_maps.shape[3]
        best_pos = int(pos_per_angle[best_angle_idx])
        y = best_pos // score_w
        x = best_pos % score_w

        results.append((best_score, x, y, best_angle_idx))

    return results


def _batched_match(
    image: FloatArray2D,
    templates: list[np.ndarray],
    angles: np.ndarray,
    minimum_fill_fraction: float,
    fill_value: float,
    chunk_size: int = 20,
) -> list[tuple[float, int, int, int]]:
    """Full-resolution matching using batched GPU operations, chunked to avoid OOM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best: list[tuple[float, int, int, int]] = [(-np.inf, 0, 0, 0) for _ in templates]

    for chunk_start in range(0, len(angles), chunk_size):
        angle_chunk = angles[chunk_start : chunk_start + chunk_size]
        batch, valid_batch, _ = _prepare_rotated_batch(image, angle_chunk, fill_value)

        batch_gpu = torch.from_numpy(batch).to(device)
        valid_gpu = torch.from_numpy(valid_batch).to(device)

        try:
            chunk_results = _ncc(
                batch_gpu,
                valid_gpu,
                templates,
                minimum_fill_fraction,
                device,
            )
        finally:
            del batch_gpu, valid_gpu
            torch.cuda.empty_cache()

        for i, (score, x, y, local_angle_idx) in enumerate(chunk_results):
            if score > best[i][0]:
                best[i] = (score, x, y, chunk_start + local_angle_idx)

    return best


def _rotate_image(
    image: FloatArray2D,
    angle: float,
    fill_value: float = np.nan,
) -> FloatArray2D:
    """Rotate *image* by *angle* degrees using OpenCV (resize=True)."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos, sin_ = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_ + w * cos)
    new_h = int(h * cos + w * sin_)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return np.asarray(
        cv2.warpAffine(
            image.astype(np.float32),
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(fill_value),
        ),
        dtype=np.float64,
    )
