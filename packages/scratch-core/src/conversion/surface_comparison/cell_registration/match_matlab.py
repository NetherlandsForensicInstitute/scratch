from collections.abc import Iterable

from container_models.scan_image import ScanImage
from conversion.surface_comparison.cell_registration.utils import (
    pad_image_array,
    convert_grid_cell_to_cell,
)
from conversion.surface_comparison.models import GridCell, ComparisonParams, Cell
import numpy as np
from skimage.transform import rotate


def match_cells(
    grid_cells: Iterable[GridCell],
    comparison_image: ScanImage,
    params: ComparisonParams,
    *args,
    **kwargs,
    # Add unused args to make it compatible with the implementation in `match.py`
) -> list[Cell]:
    grid_cells = list(grid_cells)
    if not grid_cells:
        return []

    pad_width, pad_height = grid_cells[0].width // 2, grid_cells[0].height // 2
    comparison_data = pad_image_array(
        comparison_image.data, pad_width=pad_width, pad_height=pad_height
    )

    pixel_size = comparison_image.scale_x
    angles = np.arange(
        params.search_angle_min,
        params.search_angle_max,
        params.search_angle_step,
    )

    for grid_cell in grid_cells:
        n_valid_ref = int(np.sum(~np.isnan(grid_cell.cell_data)))
        cell_area = grid_cell.width * grid_cell.height
        min_overlap = max(
            int(params.minimum_fill_fraction_comparison * cell_area),
            int(n_valid_ref * (1.0 - params.cell_fill_reduction_max)),
        )
        # Hard floor: statistically need at least ~10 points for r to be meaningful
        min_overlap = max(min_overlap, 10)

        for angle in angles:
            angle = float(angle)
            rotated = rotate(comparison_data, angle=-angle, order=0, resize=False)

            score_map = _nan_aware_ncc_map(
                image=rotated,
                template=grid_cell.cell_data,
                min_overlap=min_overlap,
            )

            best_flat = int(np.argmax(score_map))
            score = float(score_map.flat[best_flat])

            if score > grid_cell.grid_search_params.score:
                row, col = np.unravel_index(best_flat, score_map.shape)
                grid_cell.grid_search_params.update(
                    score=score,
                    angle=angle,
                    top_left_x=int(col) - pad_width,
                    top_left_y=int(row) - pad_height,
                )

    return [convert_grid_cell_to_cell(c, pixel_size) for c in grid_cells]


def _nan_aware_ncc_map(
    image: np.ndarray,
    template: np.ndarray,
    min_overlap: int,
) -> np.ndarray:
    """
    NaN-aware normalized cross-correlation map, matching MATLAB's xcorr2_similarity.

    Computes the sliding-window Pearson r between `template` and every same-sized
    patch of `image`, counting only jointly valid (non-NaN) pixels at each shift.
    Positions with fewer than `min_overlap` jointly valid pixels are set to 0.0.

    Runs in O(N log N) via 6 FFT cross-correlations, all in float64.
    No NaN filling is performed — NaN pixels are excluded per-position.

    Output shape: (H - h + 1, W - w + 1), values clipped to [-1, 1], dtype float32.

    :param image: (H, W) float64 array, NaN where invalid.
    :param template: (h, w) float64 array, NaN where invalid.
    :param min_overlap: Minimum jointly valid pixels required; matches MATLAB nOverlapMin.
    :returns: Score map as float32.
    """
    th, tw = template.shape

    T_valid = ~np.isnan(template)
    I_valid = ~np.isnan(image)

    # Zero-substitute: NaN → 0 so FFT arithmetic is well-defined.
    # The valid masks ensure NaN pixels never contribute to any sum.
    T = np.where(T_valid, template, 0.0)
    I = np.where(I_valid, image, 0.0)  # noqa
    Tm = T_valid.astype(np.float64)  # 1 where template is valid, 0 elsewhere
    Im = I_valid.astype(np.float64)  # 1 where image is valid, 0 elsewhere

    from scipy.signal import correlate

    def _xcorr_valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Slide b (template-sized) over a (image-sized) — true cross-correlation,
        valid region only.
        correlate(a, b, mode='valid')[r, c] = sum_{i,j} a[r+i, c+j] * b[i, j]
        """
        return correlate(a, b, mode="valid", method="fft")

    # Jointly valid pixel count at each shift — this is MATLAB's `mw`
    n = _xcorr_valid(Im, Tm)

    # All sums are restricted to jointly valid pixels by the mask multiplication
    sum_T = _xcorr_valid(Im, T)  # sum of T-values where both valid
    sum_I = _xcorr_valid(I, Tm)  # sum of I-values where both valid
    sum_T2 = _xcorr_valid(Im, T**2)  # sum of T^2 where both valid
    sum_I2 = _xcorr_valid(I**2, Tm)  # sum of I^2 where both valid
    sum_TI = _xcorr_valid(I, T)  # sum of T*I where both valid

    # Pearson r = (n*ΣTI - ΣT*ΣI) / sqrt((n*ΣT² - ΣT²)(n*ΣI² - ΣI²))
    numerator = n * sum_TI - sum_T * sum_I
    var_T = n * sum_T2 - sum_T**2
    var_I = n * sum_I2 - sum_I**2

    # Clamp numerical noise that pushes variances slightly below zero
    np.maximum(var_T, 0.0, out=var_T)
    np.maximum(var_I, 0.0, out=var_I)
    denom = np.sqrt(var_T * var_I)

    # MATLAB equivalent: mc(mw < nOverlapMin) = NaN → we use 0.0
    # denom == 0 means locally flat patch → score 0.0, not spurious 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(
            (n >= min_overlap) & (denom > 0.0),
            numerator / denom,
            0.0,
        )

    return np.clip(score, -1.0, 1.0).astype(np.float32)
