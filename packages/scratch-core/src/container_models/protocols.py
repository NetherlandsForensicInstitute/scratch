from typing import Protocol
from numpy.typing import NDArray


class DesignMatrixSolver(Protocol):
    """Protocol for solver functions that compute polynomial coefficients from a design matrix and Z values."""

    def __call__(self, design_matrix: NDArray, zs: NDArray) -> NDArray: ...
