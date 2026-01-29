from enum import Flag, auto
import numpy as np
from typing import Callable
from pydantic import BaseModel

from container_models.base import FloatArray1D, FloatArray2D


class SurfaceTerms(Flag):
    """
    Surface terms to include in the leveling fit.

    Terms can be combined using bitwise OR, e.g.: OFFSET | TILT_X | TILT_Y
    """

    OFFSET = auto()  # Constant offset (z = c)
    TILT_X = auto()  # Linear tilt in X (z = ax)
    TILT_Y = auto()  # Linear tilt in Y (z = by)
    ASTIG_45 = auto()  # 45-degree astigmatism (z = cxy)
    DEFOCUS = auto()  # Defocus/sphere (z = x² + y²)
    ASTIG_0 = auto()  # 0-degree astigmatism (z = x² - y²)

    # Common presets
    NONE = 0
    PLANE = OFFSET | TILT_X | TILT_Y
    SPHERE = OFFSET | TILT_X | TILT_Y | ASTIG_45 | DEFOCUS | ASTIG_0


# Mapping mathematical term indices to lambda functions for design matrix generation
TERM_FUNCTIONS: dict[
    SurfaceTerms, Callable[[FloatArray1D, FloatArray1D], FloatArray1D]
] = {
    SurfaceTerms.OFFSET: lambda xs, ys: np.ones_like(xs),
    SurfaceTerms.TILT_X: lambda xs, ys: xs,
    SurfaceTerms.TILT_Y: lambda xs, ys: ys,
    SurfaceTerms.ASTIG_45: lambda xs, ys: xs * ys,
    SurfaceTerms.DEFOCUS: lambda xs, ys: xs**2 + ys**2,
    SurfaceTerms.ASTIG_0: lambda xs, ys: xs**2 - ys**2,
}


class LevelingResult(BaseModel, arbitrary_types_allowed=True):
    """
    Result of a leveling operation.

    :param leveled_map: 2D array with the leveled height data
    :param parameters: Dictionary mapping SurfaceTerms to fitted coefficient values
    :param residual_rms: Root mean square of residuals after leveling
    :param fitted_surface: 2D array of the fitted surface (same shape as `leveled_map`)
    """

    leveled_map: FloatArray2D
    parameters: dict[SurfaceTerms, float]
    residual_rms: float
    fitted_surface: FloatArray2D


class NormalizedCoordinates(BaseModel, arbitrary_types_allowed=True):
    """
    Container class for storing centered and rescaled coordinates.

    :param xs: The rescaled X-coordinates.
    :param ys: The rescaled Y-coordinates.
    :param x_mean: The mean of the X-coordinates before rescaling.
    :param y_mean: The mean of the Y-coordinates before rescaling.
    :param scale: The multiplier used for rescaling.
    """

    xs: FloatArray1D
    ys: FloatArray1D
    x_mean: float
    y_mean: float
    scale: float
