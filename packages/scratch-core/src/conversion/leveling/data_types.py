from enum import Flag, auto
from pydantic import BaseModel

from container_models.base import FloatArray2D


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


class LevelingResult(BaseModel, arbitrary_types_allowed=True):
    """
    Result of a leveling operation.

    :param leveled_map: 2D array with the leveled height data
    :param fitted_surface: 2D array of the fitted surface (same shape as `leveled_map`)
    """

    leveled_map: FloatArray2D
    fitted_surface: FloatArray2D
