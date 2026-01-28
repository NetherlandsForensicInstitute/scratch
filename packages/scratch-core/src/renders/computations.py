from itertools import product
from numpy.typing import NDArray

from container_models.base import Point
import numpy as np


def clip_factors(factors: Point[float], preserve_aspect_ratio: bool) -> Point[float]:
    """Clip the scaling factors to minimum 1.0, while keeping the aspect ratio if `preserve_aspect_ratio` is True."""
    if preserve_aspect_ratio:
        max_factor = max(factors.x, factors.y)
        factors = Point(max_factor, max_factor)

    return Point(max(factors.x, 1.0), max(factors.y, 1.0))


def generate_polynomial_exponents(order: int) -> list[tuple[int, int]]:
    """
    Generate polynomial exponent pairs for 2D polynomial terms up to a given order.

    :param order: Maximum total degree (py + px) for the polynomial terms.
    :returns: List of (power_y, power_x) tuples representing polynomial terms.
    """
    return [
        (py, px) for py, px in product(range(order + 1), repeat=2) if py + px <= order
    ]


def solve_least_squares(
    design_matrix: NDArray[np.float64], zs: NDArray
) -> NDArray[np.floating]:
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
