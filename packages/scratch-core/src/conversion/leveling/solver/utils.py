import numpy as np
from typing import Any
from numpy.typing import NDArray


def compute_root_mean_square(data: NDArray[Any]) -> float:
    """Compute the root-mean-square from a data array and return as Python float."""
    return float(np.sqrt(np.nanmean(data**2)))
