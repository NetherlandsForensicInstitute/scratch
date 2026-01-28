from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class DesignMatrixSolver(Protocol):
    """Protocol for solver functions that compute polynomial coefficients from a design matrix and Z values."""

    def __call__[T: np.number](
        self, design_matrix: NDArray[T], zs: NDArray
    ) -> NDArray[T]: ...
