import numpy as np
from container_models.base import ScanMap2DArray
from container_models.scan_image import ScanImage
from conversion.leveling.data_types import SurfaceTerms
from renders.computations import solve_least_squares
from renders.spacial import generate_point_cloud, fit_surface


def level_map(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    reference_point: tuple[float, float],
) -> ScanMap2DArray:
    """
    Compute the leveled map by fitting polynomial terms and subtracting them from the image data.

    This computation effectively acts as a high-pass filter on the image data.

    :param scan_image: The scan image containing the image data to level.
    :param terms: The surface terms to use in the polynomial fitting. Can be combined using bit-operators.
    :param reference_point: Tuple (X, Y) in physical coordinate space defining the origin for coordinate translation.
        The coordinates will be translated such that (X, Y) lies at the origin.
    :returns: 2D array containing the leveled scan data (original data minus fitted surface).
    """
    point_cloud = generate_point_cloud(
        scan_image=scan_image, reference_point=reference_point
    )
    fitted_surface = fit_surface(
        point_cloud=point_cloud, terms=terms, solver=solve_least_squares
    )
    leveled_map_2d = np.full_like(scan_image.data, np.nan)
    leveled_map_2d[scan_image.valid_mask] = point_cloud.zs - fitted_surface

    return leveled_map_2d
