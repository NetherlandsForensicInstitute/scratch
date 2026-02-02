import numpy as np

from conversion.container_models.base import FloatArray1D
from conversion.leveling import SurfaceTerms
from conversion.leveling.solver import (
    normalize_coordinates,
    build_design_matrix,
    denormalize_parameters,
)


def fit_surface(
    xs: FloatArray1D, ys: FloatArray1D, zs: FloatArray1D, terms: SurfaceTerms
) -> tuple[FloatArray1D, dict[SurfaceTerms, float]]:
    """
    Core solver: fits a surface to the point cloud (xs, ys, zs).

    :param xs: The X-coordinates.
    :param ys: The Y-coordinates.
    :param zs: The Z-values.
    :param terms: The terms to use in the fitting
    :return: A tuple containing the fitted surface (z̃s) and the estimated physical parameters.
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
        dict(zip(terms, map(float, coefficients))),
        normalized.x_mean,
        normalized.y_mean,
        normalized.scale,
    )
    return fitted_surface, physical_params
