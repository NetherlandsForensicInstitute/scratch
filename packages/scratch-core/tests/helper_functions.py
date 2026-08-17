from functools import partial
from typing import Any

import numpy as np

from container_models.base import DepthData
from container_models.scan_image import ScanImage
from conversion.data_formats import MarkImpressionType, MarkType, Mark
from conversion.surface_comparison.models import Cell, CellMetaData


def assert_nan_mask_preserved(
    input_array: np.ndarray, output_array: np.ndarray
) -> None:
    """Assert that NaN locations in input are still NaN in output, and vice versa."""
    if input_array.ndim > output_array.ndim:
        input_nan = np.isnan(input_array[..., 0])
    else:
        input_nan = np.isnan(input_array)

    if output_array.ndim > input_array.ndim:
        output_nan = np.isnan(output_array[..., 0])
    else:
        output_nan = np.isnan(output_array)

    np.testing.assert_array_equal(input_nan, output_nan)


def make_mark(
    data: DepthData,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    mark_type: MarkType = MarkImpressionType.EXTRACTOR_IMPRESSION,
    meta_data: dict[str, Any] | None = None,
) -> Mark:
    """Create a Mark instance for testing."""
    scan_image = ScanImage(data=data, scale_x=scale_x, scale_y=scale_y)
    if meta_data is not None:
        return Mark(
            scan_image=scan_image,
            mark_type=mark_type,
            meta_data=meta_data,
        )
    return Mark(scan_image=scan_image, mark_type=mark_type)


def make_cell(
    center_reference: tuple[float, float] = (0.0, 0.0),
    best_score: float = 0.8,
    is_congruent: bool = False,
    angle_deg: float = 0.0,
    center_comparison: tuple[float, float] | None = None,
    cell_size: tuple[float, float] = (50e-6, 50e-6),
    fill_fraction_reference: float = 0.9,
    is_outlier: bool | None = None,
    residual_angle_deg: float = 0.0,
    position_error: tuple[float, float] = (0.0, 0.0),
) -> Cell:
    """Create a Cell instance for testing."""
    if center_comparison is None:
        center_comparison = center_reference
    if is_outlier is None:
        is_outlier = not is_congruent

    return Cell(
        center_reference=center_reference,
        cell_size=cell_size,
        fill_fraction_reference=fill_fraction_reference,
        best_score=best_score,
        angle_deg=angle_deg,
        center_comparison=center_comparison,
        is_congruent=is_congruent,
        meta_data=CellMetaData(
            is_outlier=is_outlier,
            residual_angle_deg=residual_angle_deg,
            position_error=position_error,
        ),
    )


NoScaleScanImage = partial(ScanImage, scale_x=1, scale_y=1)
