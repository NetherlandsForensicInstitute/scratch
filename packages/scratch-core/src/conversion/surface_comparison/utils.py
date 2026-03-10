import numpy as np
from numpy.typing import NDArray


def convert_meters_to_pixels(
    values: tuple[float, float], pixel_size: float
) -> tuple[int, int]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> int:
        return int(round(value / pixel_size))

    return _convert(values[0]), _convert(values[1])


def convert_pixels_to_meters(
    values: tuple[float, float], pixel_size: float
) -> tuple[float, float]:
    """TODO: Remove this function if possible."""

    def _convert(value: float) -> float:
        return value * pixel_size

    return _convert(values[0]), _convert(values[1])


def compute_fill_fraction(array: NDArray) -> float:
    """Compute the fraction of valid (non-NaN) values in the array."""
    return float(np.count_nonzero(~np.isnan(array)) / array.size)
