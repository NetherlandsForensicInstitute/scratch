from numpy.typing import NDArray

import numpy as np


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
