import numpy as np
from conversion.leveling import SurfaceTerms, LevelingResult
from conversion.leveling.solver import (
    fit_surface,
    get_2d_grid,
    compute_root_mean_square,
)
from conversion.leveling.solver.utils import compute_image_center
from container_models.scan_image import ScanImage


def level_map(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    reference_point: tuple[float, float] | None = None,
) -> LevelingResult:
    """
    Compute the leveled map by fitting polynomial terms and subtracting them from the image data.

    This computation effectively acts as a high-pass filter on the image data.

    :param scan_image: The scan image containing the image data to level.
    :param terms: The surface terms to use in the fitting. Note: terms can be combined using bit-operators.
    :param reference_point: A tuple representing a reference point (X, Y) in physical coordinate space.
        If provided, then the coordinates will be translated such that (X, Y) lies in the origin after translation.
        If `None`, then the coordinates will be translated such that the center of the image lies in the origin.
    :returns: An instance of `LevelingResult` containing the leveled scan data and estimated physical parameters.
    """
    if not reference_point:
        reference_point = compute_image_center(scan_image)

    # Build the 2D grids and translate in the opposite direction of `reference_point`
    x_grid, y_grid = get_2d_grid(
        scan_image, offset=(-reference_point[0], -reference_point[1])
    )

    # Get the point cloud (xs, ys, zs) for the numerical data
    xs, ys, zs = (
        x_grid[scan_image.valid_mask],
        y_grid[scan_image.valid_mask],
        scan_image.valid_data,
    )

    # Fit surface by solving the least-squares solution to a linear matrix equation
    fitted_surface, physical_params = fit_surface(xs, ys, zs, terms)
    fitted_surface_2d = np.full_like(scan_image.data, np.nan)
    fitted_surface_2d[scan_image.valid_mask] = fitted_surface

    # Compute the leveled map
    leveled_map_2d = np.full_like(scan_image.data, np.nan)
    leveled_map_2d[scan_image.valid_mask] = zs - fitted_surface

    # Calculate RMS of residuals
    residual_rms = compute_root_mean_square(leveled_map_2d)

    # TODO:
    # Why LevelingResult return
    # I looked for all references of this funciton and this object
    # But it is only refernce twice and in both cases we only care about leveled_map (leveled_map_2d)
    # Thus In my perspective this can be nuked
    return LevelingResult(
        leveled_map=leveled_map_2d,
        parameters=physical_params,
        residual_rms=residual_rms,
        fitted_surface=fitted_surface_2d,
    )
