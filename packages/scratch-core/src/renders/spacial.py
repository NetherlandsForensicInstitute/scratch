from __future__ import annotations
from container_models.base import Point, PointCloud
from conversion.leveling.solver.grid import get_2d_grid
import numpy as np
from typing import TYPE_CHECKING

from container_models.protocols import DesignMatrixSolver
from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.transforms import normalize_coordinates

if TYPE_CHECKING:
    from container_models.scan_image import ScanImage
    from numpy.typing import NDArray


def generate_point_cloud(
    scan_image: ScanImage, reference_point: Point[float]
) -> PointCloud:
    """
    Generate a 3D point cloud from a scan image with coordinates centered at a reference point.

    :param scan_image: The scan image containing the height data and mask.
    :param reference_point: Tuple (x, y) in physical coordinates to use as the origin.
    :returns: PointCloud containing the valid X, Y, and Z coordinates.
    """
    # Build the 2D grids and translate in the opposite direction of `reference_point`
    x_grid, y_grid = get_2d_grid(
        scan_image, offset=(-reference_point.x, -reference_point.y)
    )
    # Get the point cloud (xs, ys, zs) for the numerical data
    xs, ys, zs = (
        x_grid[scan_image.valid_mask],
        y_grid[scan_image.valid_mask],
        scan_image.valid_data,
    )
    return PointCloud(xs=xs, ys=ys, zs=zs)


def fit_surface(
    point_cloud: PointCloud, terms: SurfaceTerms, solver: DesignMatrixSolver
) -> NDArray[np.float64]:
    """
    Core solver: fits a surface to the point cloud.

    :param point_cloud: PointCloud containing the X, Y, and Z coordinates.
    :param terms: The surface terms to use in the polynomial fitting.
    :param solver: Solver function that computes coefficients from design matrix and Z values.
    :returns: 1D array containing the fitted surface values (z̃s).
    """
    normalized = normalize_coordinates(point_cloud.xs, point_cloud.ys)
    design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)
    coefficients = solver(design_matrix=design_matrix, zs=point_cloud.zs)
    return design_matrix @ coefficients
