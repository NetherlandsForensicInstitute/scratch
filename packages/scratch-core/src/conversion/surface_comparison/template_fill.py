import numpy as np

from container_models.base import FloatArray2D


def fill_template_nan(
    array: FloatArray2D, nan_fill_value: float | None = None
) -> FloatArray2D:
    """
    Replace NaN in a reference template, per the configured fill strategy.

    ``None`` (the default) uses the array's own valid-pixel mean, falling back to 0.0 when no
    pixel is valid; any other value is used as-is. This is the *only* implementation of that
    rule - coarse and full-resolution templates must both route through it, because the coarse
    stage is what chooses each cell's candidate locations, so a different fill there changes the
    search result rather than just the refinement.

    Why the local mean matters: once ``_prepare_templates`` centers a template on its own mean, a
    template filled with that same mean has every filled pixel land on exactly zero, so missing
    pixels drop out of the correlation, the norm and the variance entirely. A scene-wide fill
    value leaves them non-zero, i.e. missing data contributes to the score as though it were real,
    flat surface at that height.

    This lives in its own module, depending on nothing inside ``surface_comparison``, because both
    ``models`` and ``cell_registration.utils`` need it at module level and both sit above the
    other candidate homes: ``surface_comparison.utils`` imports ``Cell``, and
    ``cell_registration.utils`` imports ``Cell``/``GridCell``, so either would close a cycle.

    :param array: Input 2D array, possibly containing NaN.
    :param nan_fill_value: Explicit fill value, or ``None`` for the array's own valid-pixel mean.
    :returns: Copy of *array* with NaN replaced.
    """
    if nan_fill_value is None:
        local_mean = np.nanmean(array)
        nan_fill_value = float(local_mean) if np.isfinite(local_mean) else 0.0
    return np.nan_to_num(array, nan=nan_fill_value, copy=True)
