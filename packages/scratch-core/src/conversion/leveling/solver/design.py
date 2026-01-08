import numpy as np
from numpy.typing import NDArray
from conversion.leveling import SurfaceTerms
from conversion.leveling.data_types import TERM_FUNCTIONS


def build_design_matrix(
    xs: NDArray, ys: NDArray, terms: SurfaceTerms
) -> NDArray[np.float64]:
    """
    Constructs the Least Squares design matrix based on grid coordinates (xs, ys) and requested terms.

    :param xs: The X-coordinates.
    :param ys: The Y-coordinates.
    :param terms: The surface terms to use in the design matrix.
    :returns: The design matrix as a numpy array with shape [n_points, n_terms].
    """
    num_points = xs.size
    matrix = np.zeros((num_points, len(terms)), dtype=np.float64)

    for column_index, term in enumerate(terms):
        if func := TERM_FUNCTIONS.get(term):
            matrix[:, column_index] = func(xs, ys)

    return matrix
