import numpy as np
from computations.constants import SurfaceTerms
from container_models.models import LevelingResult
from container_models.scan_image import ScanImage

from surfalize import Surface
from container_models.base import BinaryMask


def get_bounding_box(mask: BinaryMask, margin: int) -> tuple[slice, slice]:
    """
    Compute the minimal bounding box of a 2D mask.

    Finds the smallest axis-aligned rectangle containing all non-zero (or True) values.

    :param mask: 2D mask (non-zero/True values indicate the region of interest)
    :param margin: Margin around the bounding box to either crop (positive) or extend (negative) the bounding box
    :returns: Tuple (y_slice, x_slice) as slices for bounding_box.
    """
    y_coords, x_coords = np.nonzero(mask)
    y_min = max(0, y_coords.min() + margin)
    y_max = min(mask.shape[0], y_coords.max() - margin + 1)
    x_min = max(0, x_coords.min() + margin)
    x_max = min(mask.shape[1], x_coords.max() - margin + 1)

    if x_min >= x_max:
        raise ValueError("Slice results in x_min >= x_max. Margin may be too large.")
    if y_min >= y_max:
        raise ValueError("Slice results in y_min >= y_max. Margin may be too large.")

    return slice(y_min, y_max), slice(x_min, x_max)


def level_map(scan_image: ScanImage, terms: SurfaceTerms) -> LevelingResult:
    """
    Compute the leveled map by fitting polynomial terms and subtracting them from the image data.

    This computation effectively acts as a high-pass filter on the image data.

    :param scan_image: The scan image containing the image data to level.
    :param terms: The surface terms to use in the fitting. Note: terms can be combined using bit-operators.
    :returns: An instance of `LevelingResult` containing the leveled scan data and estimated physical parameters.
    """
    if terms == SurfaceTerms.NONE:
        return LevelingResult(
            leveled_map=scan_image.data,
            fitted_surface=np.full_like(scan_image.data, 0.0),
        )
    polynomial_degree = terms  # semantic renaming of enum value
    if scan_image.valid_mask.sum() < 1 + polynomial_degree:
        raise ValueError(
            f"At least {1 + polynomial_degree} values are needed for the least squares solver."
        )
    surface = Surface(
        height_data=scan_image.data,
        step_x=scan_image.scale_x,
        step_y=scan_image.scale_y,
    )
    leveled, trend = surface.detrend_polynomial(
        degree=polynomial_degree, inplace=False, return_trend=True
    )
    return LevelingResult(leveled_map=leveled.data, fitted_surface=trend.data)
