import numpy as np
from numpy.typing import NDArray
from collections.abc import Mapping
from conversion.leveling import SurfaceTerms
from conversion.leveling.data_types import RescaledCoordinatesResult


def center_and_scale_coordinates(
    x_grid: NDArray, y_grid: NDArray
) -> RescaledCoordinatesResult:
    """
    Center and scale grid coordinates.

    This method is used to improve numerical stability during fitting.

    :param x_grid: The X-coordinates of the grid to normalize.
    :param y_grid: The Y-coordinates of the grid to normalize.
    :returns: An instance of `NormalizedCoordinatesResult` containing the rescaled grid coordinates.
    """
    x_mean, y_mean = np.mean(x_grid), np.mean(y_grid)
    vx_norm = x_grid - x_mean
    vy_norm = y_grid - y_mean

    span_x = np.max(vx_norm) - np.min(vx_norm)
    span_y = np.max(vy_norm) - np.min(vy_norm)
    # Avoid division by zero
    max_span = max(span_x, span_y)
    scale = 1.0 if np.isclose(max_span, 0.0) else 1 / max_span

    return RescaledCoordinatesResult(
        x_grid=vx_norm * scale,
        y_grid=vy_norm * scale,
        x_mean=float(x_mean),
        y_mean=float(y_mean),
        scale=float(scale),
    )


def denormalize_parameters(
    coefficients: Mapping[SurfaceTerms, NDArray],
    x_mean: float,
    y_mean: float,
    scale: float,
) -> dict[SurfaceTerms, float]:
    """
    Converts normalized fit parameters back to real-world physical units.

    The computation matches the specific numerical corrections from the original MATLAB script.

    :param coefficients: A dictionary containing the normalized fit parameters.
    :param x_mean: The mean of the x-coordinates.
    :param y_mean: The mean of the y-coordinates.
    :param scale: The scale factor.
    :returns: A dictionary containing the denormalized fit parameters for all surface terms.
    """
    params = np.array(
        [coefficients.get(term, 0.0) for term in SurfaceTerms], dtype=np.float64
    )

    # Un-normalize scaling
    params[1:3] *= scale  # Tilts
    params[3:] *= scale**2  # Quadratic terms

    # Algebraic corrections for centering (x_mean, y_mean)
    # Note: These formulas correspond exactly to the MATLAB implementation
    # P[0] = Offset, P[1] = TiltX, P[2] = TiltY, etc.

    # Adjust Offset (p0)
    params[0] = (
        params[0]
        - params[1] * x_mean
        - params[2] * y_mean
        + params[3] * x_mean * y_mean
        + params[4] * (x_mean**2 + y_mean**2)
        + params[5] * (x_mean**2 - y_mean**2)
    )
    # Adjust Tilt X (p1)
    params[1] = (
        params[1] - params[3] * y_mean - 2 * params[4] * x_mean - 2 * params[5] * x_mean
    )
    # Adjust Tilt Y (p2)
    params[2] = (
        params[2] - params[3] * x_mean - 2 * params[4] * y_mean + 2 * params[5] * y_mean
    )

    return dict(zip(SurfaceTerms, map(float, params)))
