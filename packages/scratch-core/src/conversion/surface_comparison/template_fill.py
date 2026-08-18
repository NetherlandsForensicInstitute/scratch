import numpy as np

from container_models.base import FloatArray2D


def fill_template_nan(
    array: FloatArray2D, nan_fill_value: float | None = None
) -> FloatArray2D:
    """
    Replace NaN in a reference template using the specified fill strategy.

    If ``nan_fill_value`` is ``None``, the array's valid-pixel mean is used (falling back to 0.0).
    Using the local mean ensures filled pixels contribute minimally to the score after centering.

    :param array: Input 2D array, possibly containing NaN.
    :param nan_fill_value: Explicit fill value, or ``None`` for the array's own valid-pixel mean.
    :returns: Copy of *array* with NaN replaced.
    """
    if nan_fill_value is None:
        local_mean = np.nanmean(array)
        nan_fill_value = float(local_mean) if np.isfinite(local_mean) else 0.0
    return np.nan_to_num(array, nan=nan_fill_value, copy=True)
