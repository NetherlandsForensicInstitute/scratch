"""Helper functions for conversion tests."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from matplotlib.figure import Figure

from container_models.base import FloatArray, DepthData
from container_models.scan_image import ScanImage
from conversion.data_formats import Mark, MarkType

from numpy.typing import NDArray


def _crop_to_common_shape(
    arr1: NDArray, arr2: NDArray, center_crop: bool = False
) -> tuple[NDArray, NDArray]:
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


def _compute_correlation(arr1: FloatArray, arr2: FloatArray) -> float:
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


def _compute_difference_stats(arr1: FloatArray, arr2: FloatArray) -> dict[str, float]:
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


def make_mark(
    data: DepthData,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    mark_type: MarkType = MarkType.EXTRACTOR_IMPRESSION,
    center: tuple[float, float] | None = None,
    meta_data: dict[str, Any] | None = None,
) -> Mark:
    """Create a Mark instance for testing."""
    scan_image = ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)
    if meta_data is not None:
        return Mark(
            scan_image=scan_image,
            mark_type=mark_type,
            center=center,
            meta_data=meta_data,
        )
    return Mark(scan_image=scan_image, mark_type=mark_type, center=center)


def assert_plot_is_valid_image(fig: Figure, tmp_path: Path) -> None:
    img_path = tmp_path / "plot.png"

    fig.savefig(img_path, format="png")

    assert img_path.exists()
    assert img_path.stat().st_size > 0

    # Validate it's a real image
    img = Image.open(img_path)
    img.verify()
