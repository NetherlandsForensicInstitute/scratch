import numpy as np
from conversion.leveling import SurfaceTerms, LevelingResult
from conversion.leveling.solver import (
    fit_surface,
    prepare_2d_grid,
    compute_root_mean_square,
)
from image_generation.data_formats import ScanImage


def level_map(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    image_center: tuple[float, float] | None = None,
) -> LevelingResult:
    """
    Compute the leveled map by fitting polynomial terms to the image data and subtracting them from the image data.

    This computation effectively acts as a high-pass filter on the image data.

    :param scan_image: The scan image containing the image data to level.
    :param terms: The surface terms to use in the fitting. Note: terms can be combined using the bit-operators.
    :param image_center: The center of the image to use for rescaling the grid coordinates.
    :returns: An instance of `LevelingResult` containing the leveled scan data and estimated physical parameters.
    """
    # Build the 2D grids
    x_grid, y_grid = prepare_2d_grid(scan_image, image_center=image_center)
    valid_mask = ~np.isnan(scan_image.data)

    # Get the point cloud (xs, ys, zs) for the numerical data
    xs, ys, zs = (
        x_grid[valid_mask],
        y_grid[valid_mask],
        scan_image.data[valid_mask],
    )

    # Fit surface by solving the least-squares solution to a linear matrix equation
    result = fit_surface(xs, ys, zs, terms)
    fitted_surface_2d = np.full_like(scan_image.data, np.nan)
    fitted_surface_2d[valid_mask] = result.fitted_surface

    # Compute the leveled map
    leveled_map_2d = np.full_like(scan_image.data, np.nan)
    leveled_map_2d[valid_mask] = zs - result.fitted_surface

    # Calculate RMS of residuals
    residual_rms = compute_root_mean_square(leveled_map_2d)

    return LevelingResult(
        leveled_map=leveled_map_2d,
        parameters=result.physical_params,
        residual_rms=residual_rms,
        fitted_surface=fitted_surface_2d,
    )
