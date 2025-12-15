import numpy as np
from numpy.typing import NDArray
from collections.abc import Mapping
from conversion.leveling import SurfaceTerms, TERM_FUNCTIONS, LevelingResult
from image_generation.data_formats import ScanImage


def _prepare_2d_grid(
    scan_image: ScanImage, image_center: tuple[float, float] | None = None
) -> tuple[NDArray, NDArray]:
    """
    TODO: write a docstring
    """
    # Generate Grid (ij indexing to match matrix coordinates)
    x_indices, y_indices = np.meshgrid(
        np.arange(scan_image.width), np.arange(scan_image.height), indexing="ij"
    )
    # Center grid
    if not image_center:
        image_center = (
            (scan_image.width - 1)
            * scan_image.scale_x
            * 0.5,  # X-coordinate of image center
            (scan_image.height - 1)
            * scan_image.scale_y
            * 0.5,  # Y-coordinate of image center
        )
    x_grid = (x_indices * scan_image.scale_x) - image_center[0]
    y_grid = (y_indices * scan_image.scale_y) - image_center[1]

    return x_grid, y_grid


def _build_design_matrix(
    x_grid: NDArray, y_grid: NDArray, terms: SurfaceTerms
) -> NDArray:
    """
    Constructs the Least Squares design matrix based on requested terms.
    """
    num_points = x_grid.size
    matrix = np.zeros((num_points, len(terms)), dtype=np.float64)

    for column_index, term in enumerate(terms):
        if func := TERM_FUNCTIONS.get(term):
            matrix[:, column_index] = func(x_grid, y_grid)

    return matrix


def _normalize_coordinates(
    x_grid: NDArray, y_grid: NDArray
) -> tuple[NDArray, NDArray, float, float, float]:
    """
    Centers and scales coordinates to improve numerical stability during fitting.
    """
    x_mean, y_mean = np.mean(x_grid), np.mean(y_grid)
    vx_norm = x_grid - x_mean
    vy_norm = y_grid - y_mean

    span_x = np.max(vx_norm) - np.min(vx_norm)
    span_y = np.max(vy_norm) - np.min(vy_norm)
    # Avoid division by zero
    max_span = max(span_x, span_y)
    scale = 1.0 / max_span if max_span > 0 else 1.0

    return vx_norm * scale, vy_norm * scale, float(x_mean), float(y_mean), float(scale)


def _denormalize_parameters(
    coefficients: Mapping[SurfaceTerms, NDArray], x_mean: float, y_mean: float, s: float
) -> dict[SurfaceTerms, float]:
    """
    Converts normalized fit parameters back to real-world physical units.
    Matches the specific algebraic logic from the original MATLAB script.

    TODO: explain params
    """
    params = np.array(
        [coefficients.get(term, 0.0) for term in SurfaceTerms], dtype=np.float64
    )

    # Un-normalize scaling
    params[1:3] *= s  # Tilts
    params[3:] *= s**2  # Quadratic terms

    # Algebraic corrections for centering (x_mean, y_mean)
    # Note: These formulas correspond exactly to the MATLAB implementation
    # P[0] = Offset, P[1] = TiltX, P[2] = TiltY, etc.

    # Adjust Offset (p0)
    params[0] = (
        params[0]
        - params[1] * x_mean
        - params[2] * y_mean
        + params[3] * x_mean * y_mean
        + params[4] * (x_mean**2 + y_mean**2)
        + params[5] * (x_mean**2 - y_mean**2)
    )
    # Adjust Tilt X (p1)
    params[1] = (
        params[1] - params[3] * y_mean - 2 * params[4] * x_mean - 2 * params[5] * x_mean
    )
    # Adjust Tilt Y (p2)
    params[2] = (
        params[2] - params[3] * x_mean - 2 * params[4] * y_mean + 2 * params[5] * y_mean
    )

    return dict(zip(SurfaceTerms, map(float, params)))


def _solve_leveling(
    x_grid: NDArray, y_grid: NDArray, z_grid: NDArray, terms: SurfaceTerms
) -> tuple[NDArray, dict[SurfaceTerms, float]]:
    """
    Core solver: fits a surface to the point cloud (xs, ys, zs).

    :param x_grid: X coordinates
    :param y_grid: Y coordinates
    :param z_grid: Z values
    :param terms: the terms to use in the fitting
    :return: Tuple (fitted surface, physical parameters).
    """
    # 1. Normalize
    x_normalized, y_normalized, x_mean, y_mean, scale = _normalize_coordinates(
        x_grid, y_grid
    )
    # 2. Build Matrix
    design_matrix = _build_design_matrix(x_normalized, y_normalized, terms)
    # 3. Solve (Least Squares)
    (
        coefficients,
        *_,
    ) = np.linalg.lstsq(design_matrix, z_grid, rcond=None)
    # 4. Calculate deviations
    fitted_surface = design_matrix @ coefficients
    # 5. Recover physical parameters (optional usage, but part of original spec)
    physical_params = _denormalize_parameters(
        dict(zip(terms, coefficients)), x_mean, y_mean, scale
    )

    return fitted_surface, physical_params


def _compute_root_mean_square(data: NDArray) -> float:
    return float(np.sqrt(np.mean(data**2)))


def level_map(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    is_highpass: bool = True,
    image_center: tuple[float, float] | None = None,
):
    """
    TODO: write a docstring
    """
    # Build grids
    x_grid, y_grid = _prepare_2d_grid(scan_image, image_center=image_center)
    valid_mask = ~np.isnan(scan_image.data)
    x_grid, y_grid, z_grid = (
        x_grid[valid_mask],
        y_grid[valid_mask],
        scan_image.data[valid_mask],
    )

    # Solve
    fitted_surface, physical_params = _solve_leveling(x_grid, y_grid, z_grid, terms)

    # Compute leveled map
    z_leveled = z_grid - fitted_surface if is_highpass else fitted_surface
    leveled_map = np.full_like(scan_image.data, np.nan)
    leveled_map[valid_mask] = z_leveled

    # Calculate RMS of residuals
    residual_rms = _compute_root_mean_square(z_leveled)

    return LevelingResult(
        leveled_map=leveled_map,
        parameters=physical_params,
        residual_rms=residual_rms,
        fitted_surface=fitted_surface,
    )
