import numpy as np


def _crop_to_common_shape(
    arr1: np.ndarray, arr2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop two arrays to their common shape."""
    min_rows = min(arr1.shape[0], arr2.shape[0])
    min_cols = min(arr1.shape[1], arr2.shape[1])
    return arr1[:min_rows, :min_cols], arr2[:min_rows, :min_cols]


def _compute_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute correlation between two arrays, ignoring NaN values."""
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    if not np.any(valid):
        return np.nan
    return np.corrcoef(arr1[valid], arr2[valid])[0, 1]


def _compute_difference_stats(arr1: np.ndarray, arr2: np.ndarray) -> dict[str, float]:
    """Compute difference statistics between two arrays."""
    diff = arr1 - arr2
    return {
        "min": float(np.nanmin(diff)),
        "max": float(np.nanmax(diff)),
        "mean": float(np.nanmean(diff)),
        "std": float(np.nanstd(diff)),
    }
