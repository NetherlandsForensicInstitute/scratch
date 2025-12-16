import numpy as np
from numpy.typing import NDArray
from conversion.leveling import SurfaceTerms
from conversion.leveling.data_types import FitSurfaceResult
from conversion.leveling.solver import (
    normalize_coordinates,
    build_design_matrix,
    denormalize_parameters,
)


def fit_surface(
    xs: NDArray, ys: NDArray, zs: NDArray, terms: SurfaceTerms
) -> FitSurfaceResult:
    """
    Core solver: fits a surface to the point cloud (xs, ys, zs).

    :param xs: The X-coordinates.
    :param ys: The Y-coordinates.
    :param zs: The Z-values.
    :param terms: The terms to use in the fitting
    :return: An instance of `FittedSurface` with the fitted surface (z̃s) and the estimated physical parameters.
    """
    # 1. Normalize the grid coordinates by centering and rescaling them
    normalized = normalize_coordinates(xs, ys)

    # 2. Build the design matrix for the least-squares solver
    design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)

    # 3. Solve (Least Squares)
    (
        coefficients,
        *_,
    ) = np.linalg.lstsq(design_matrix, zs, rcond=None)

    # 4. Compute the surface (z̃s-values) from the fitted coefficients
    fitted_surface = design_matrix @ coefficients

    # 5. Recover physical parameters (optional usage, but part of original spec)
    physical_params = denormalize_parameters(
        dict(zip(terms, coefficients)),
        normalized.x_mean,
        normalized.y_mean,
        normalized.scale,
    )

    return FitSurfaceResult(
        fitted_surface=fitted_surface, physical_params=physical_params
    )
