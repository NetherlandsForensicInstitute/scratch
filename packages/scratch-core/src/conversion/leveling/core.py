import numpy as np
from conversion.leveling import SurfaceTerms, LevelingResult

from container_models.scan_image import ScanImage

from surfalize import Surface


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

    surface = Surface(
        height_data=scan_image.data,
        step_x=scan_image.scale_x,
        step_y=scan_image.scale_y,
    )
    if terms.name == "SPHERE":
        degree = 2
    elif terms.name == "PLANE":
        degree = 1
    else:
        degree = 0
    leveled, trend = surface.detrend_polynomial(
        degree=degree, inplace=False, return_trend=True
    )
    return LevelingResult(leveled_map=leveled.data, fitted_surface=trend.data)
