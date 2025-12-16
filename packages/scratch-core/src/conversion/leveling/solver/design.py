import numpy as np
from numpy.typing import NDArray
from conversion.leveling import SurfaceTerms
from conversion.leveling.data_types import TERM_FUNCTIONS


def build_design_matrix(
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
