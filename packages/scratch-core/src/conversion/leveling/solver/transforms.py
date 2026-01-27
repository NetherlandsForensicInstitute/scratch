import numpy as np
from collections.abc import Mapping

from container_models.base import FloatArray1D
from conversion.leveling import SurfaceTerms
from conversion.leveling.data_types import NormalizedCoordinates


def normalize_coordinates(xs: FloatArray1D, ys: FloatArray1D) -> NormalizedCoordinates:
    """
    Normalize grid coordinates by centering and rescaling.

    This method is used to improve numerical stability during fitting.

    :param xs: The X-coordinates to normalize.
    :param ys: The Y-coordinates to normalize.
    :returns: An instance of `NormalizedCoordinatesResult` containing the rescaled grid coordinates.
    """
    x_mean, y_mean = np.mean(xs), np.mean(ys)
    vx_norm = xs - x_mean
    vy_norm = ys - y_mean

    span_x = np.max(vx_norm) - np.min(vx_norm)
    span_y = np.max(vy_norm) - np.min(vy_norm)
    # Avoid division by zero
    max_span = max(span_x, span_y)
    scale = 1.0 if np.isclose(max_span, 0.0) else 1 / max_span

    return NormalizedCoordinates(
        xs=vx_norm * scale,  # rescale X-coordinates
        ys=vy_norm * scale,  # rescale Y-coordinates
        x_mean=float(x_mean),
        y_mean=float(y_mean),
        scale=float(scale),
    )


def denormalize_parameters(
    coefficients: Mapping[SurfaceTerms, float],
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
