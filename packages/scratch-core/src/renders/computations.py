from itertools import product

from numpy.typing import NDArray
from conversion.filter import _convolve_2d_separable

from container_models.base import Point
import numpy as np


def generate_polynomial_exponents(order: int) -> list[Point[int]]:
    """
    Generate polynomial exponent pairs for 2D polynomial terms up to a given order.

    :param order: Maximum total degree (py + px) for the polynomial terms.
    :returns: List of (power_y, power_x) tuples representing polynomial terms.
    """
    return [
        Point(x, y) for y, x in product(range(order + 1), repeat=2) if y + x <= order
    ]


# TODO: make it generic
# def build_lhs_matrix[T: NDArray(value: T) -> T:...
# def build_lhs_matrix[T: NDArray[np.floating]](value: T) -> T:...
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
            _convolve_2d_separable(
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
