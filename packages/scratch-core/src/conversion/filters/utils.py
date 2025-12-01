import numpy as np


def _init_nan_weights(data: np.ndarray) -> np.ndarray:
    """Initialize weights with NaN locations set to zero.

    :param data: Input data array.
    :return: Weight array with 1.0 for valid data, 0.0 for NaN.
    """
    weights = np.ones_like(data)
    weights[np.isnan(data)] = 0
    return weights
