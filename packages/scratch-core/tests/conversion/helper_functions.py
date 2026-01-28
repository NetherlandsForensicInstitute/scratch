"""Helper functions for conversion tests."""

import numpy as np


def _crop_to_common_shape(
    arr1: np.ndarray, arr2: np.ndarray, center_crop: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop two arrays to their common shape.

    :param arr1: First array.
    :param arr2: Second array.
    :param center_crop: If True, extract central region when shapes differ.
                        If False, crop from top-left corner (default).
    :returns: Tuple of cropped arrays with matching shapes.
    """
    min_rows = min(arr1.shape[0], arr2.shape[0])
    min_cols = min(arr1.shape[1], arr2.shape[1])

    if center_crop:
        # Extract central region from each array
        r1_start = (arr1.shape[0] - min_rows) // 2
        c1_start = (arr1.shape[1] - min_cols) // 2
        r2_start = (arr2.shape[0] - min_rows) // 2
        c2_start = (arr2.shape[1] - min_cols) // 2

        cropped1 = arr1[r1_start : r1_start + min_rows, c1_start : c1_start + min_cols]
        cropped2 = arr2[r2_start : r2_start + min_rows, c2_start : c2_start + min_cols]
        return cropped1, cropped2
    else:
        return arr1[:min_rows, :min_cols], arr2[:min_rows, :min_cols]


def _compute_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Compute correlation between two arrays, ignoring NaN values.

    :param arr1: First array.
    :param arr2: Second array.
    :returns: Pearson correlation coefficient, or NaN if no valid values.
    """
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    if not np.any(valid):
        return np.nan
    return np.corrcoef(arr1[valid], arr2[valid])[0, 1]


def _compute_difference_stats(arr1: np.ndarray, arr2: np.ndarray) -> dict[str, float]:
    """
    Compute difference statistics between two arrays.

    :param arr1: First array.
    :param arr2: Second array.
    :returns: Dictionary with min, max, mean, and std of the difference.
    """
    diff = arr1 - arr2
    return {
        "min": float(np.nanmin(diff)),
        "max": float(np.nanmax(diff)),
        "mean": float(np.nanmean(diff)),
        "std": float(np.nanstd(diff)),
    }
