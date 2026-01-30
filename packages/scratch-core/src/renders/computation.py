from itertools import product
from typing import Protocol
from container_models.scan_image import ScanImage
from container_models.base import Point, PointCloud
from container_models.base import FloatArray2D
from conversion.filter.regression import (
    convolve_2d_separable,
    _solve_pixelwise_regression,
)

from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.transforms import normalize_coordinates
from numpy.typing import NDArray
from conversion.leveling.solver.grid import get_2d_grid
import numpy as np


def solve_least_squares(design_matrix: FloatArray2D, zs: NDArray) -> FloatArray2D:
    """
    Solve the least squares problem to find polynomial coefficients.
    :param design_matrix: The design matrix constructed from polynomial terms.
    :param zs: The Z-values (height data) to fit.
    :returns: Array of polynomial coefficients.
    """
    (
        coefficients,
        *_,
    ) = np.linalg.lstsq(design_matrix, zs, rcond=None)
    return coefficients


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


class DesignMatrixSolver(Protocol):
    """Protocol for solver functions that compute polynomial coefficients from a design matrix and Z values."""

    def __call__(self, design_matrix: NDArray, zs: NDArray) -> NDArray: ...


def fit_surface(
    point_cloud: PointCloud, terms: SurfaceTerms, solver: DesignMatrixSolver
) -> FloatArray2D:
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


def build_lhs_matrix(
    weights: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    x_coords: NDArray[np.floating],
    y_coords: NDArray[np.floating],
    exponents: list[Point[int]],
) -> NDArray[np.floating]:
    """
    Construct the LHS matrix 'A' efficiently.
    Optimization: Many terms in the matrix are identical (e.g., x*y appears multiple times).
    We compute unique moment sums first, then map them to the matrix structure.
    """
    n_params = len(exponents)

    # Calculate sum of powers for every cell in the matrix (A_pq = term_p * term_q)
    matrix_power_sums = np.array(
        [
            (exponents[p][0] + exponents[q][0], exponents[p][1] + exponents[q][1])
            for q, p in product(range(n_params), repeat=2)
        ]
    )

    # Find unique combinations of powers to avoid redundant convolutions
    unique_powers, inverse_indices = np.unique(
        matrix_power_sums, axis=0, return_inverse=True
    )

    # Compute convolutions for unique powers
    unique_moments = np.array(
        [
            convolve_2d_separable(
                weights, (x_coords**point.x) * kernel_x, (y_coords**point.y) * kernel_y
            )
            for point in unique_powers
        ]
    )

    # Reconstruct the full (n_params * n_params) matrix for every pixel
    # Current shape: (unique_terms, H, W) -> Map to -> (n_params^2, H, W)
    full_moments = unique_moments[inverse_indices]

    # Reshape to (H, W, n_params, n_params)
    # We move axes so H, W are first, creating a matrix for every pixel
    return np.moveaxis(
        full_moments.reshape(n_params, n_params, *weights.shape), [0, 1], [-2, -1]
    )


def calculate_polynomial_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    exponents: list[Point[int]],
) -> NDArray[np.floating]:
    """
    Apply Order-1 or Order-2 Local Polynomial Regression.
    This function performs a Weighted Least Squares (WLS) fit of a polynomial surface within a local window
    defined by the kernels. For each pixel, it solves the linear system A * c = b, where 'c' contains the
    coefficients of the polynomial. The smoothed value is the first coefficient (c0).
    The kernels (kernel_x, kernel_y) serve as spatial weight functions. They determine the importance of
    neighboring pixels in the regression. A non-uniform kernel (e.g., Gaussian) ensures that points closer
    to the target pixel have a higher influence on the fit than points at the window's edge, providing better
    localization and noise suppression.
    :param data: The 2D input array to be filtered. Can contain NaNs, which are treated as zero-weight during
        the regression.
    :param kernel_x: 1D array representing the horizontal weight distribution.
    :param kernel_y: 1D array representing the vertical weight distribution.
    :param exponents: List of (power_y, power_x) tuples defining the polynomial terms.
    :returns: The filtered (smoothed) version of the input data.
    """
    # 1. Setup Coordinate Systems (Normalized to [-1, 1] for stability)
    ny, nx = len(kernel_y), len(kernel_x)
    radius_y, radius_x = (ny - 1) // 2, (nx - 1) // 2

    y_coords = np.arange(-radius_y, radius_y + 1) / (radius_y if ny > 1 else 1.0)
    x_coords = np.arange(-radius_x, radius_x + 1) / (radius_x if nx > 1 else 1.0)

    # 2. Construct the Linear System Components (A matrix and b vector)
    nan_mask = np.isnan(data)
    weights = np.where(nan_mask, 0.0, 1.0)
    weighted_data = np.where(nan_mask, 0.0, data * weights)

    # Calculate RHS vector 'b' (Data Moments)
    # b_k = Convolution(weighted_data, x^px * y^py * Kernel)
    rhs_moments = np.array(
        [
            convolve_2d_separable(
                weighted_data, (x_coords**px) * kernel_x, (y_coords**py) * kernel_y
            )
            for py, px in exponents
        ]
    )

    # Calculate LHS Matrix 'A' (Weight Moments)
    # A_jk = Convolution(weights, x^(px_j + px_k) * y^(py_j + py_k) * Kernel)
    lhs_matrix = build_lhs_matrix(
        weights, kernel_x, kernel_y, x_coords, y_coords, exponents
    )

    # 3. Solve the System (A * c = b) per pixel
    return _solve_pixelwise_regression(lhs_matrix, rhs_moments, data)


def generate_polynomial_exponents(order: int) -> list[Point[int]]:
    """
    Generate polynomial exponent pairs for 2D polynomial terms up to a given order.
    :param order: Maximum total degree (py + px) for the polynomial terms.
    :returns: List of (power_y, power_x) tuples representing polynomial terms.
    """
    return [
        Point(x, y) for y, x in product(range(order + 1), repeat=2) if y + x <= order
    ]
