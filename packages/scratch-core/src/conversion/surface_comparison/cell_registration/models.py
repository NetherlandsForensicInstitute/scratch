from dataclasses import dataclass
from typing import NamedTuple

from container_models.base import FloatArray2D


@dataclass(frozen=True)
class Stage:
    """
    One resolution level's padded comparison canvas and the templates to search it with.

    :param image: Padded comparison canvas we search *in*.
    :param templates: Reference templates we search *for* (one per grid cell).
    :param fill_value: Value substituted for NaN in *image*.
    """

    image: FloatArray2D
    templates: list[FloatArray2D]
    fill_value: float


class Match(NamedTuple):
    """
    The best pose found for one cell.

    :param score: Normalized cross-correlation score.
    :param x: Left edge of the matched window, in rotated-canvas pixels.
    :param y: Top edge of the matched window, in rotated-canvas pixels.
    :param angle_deg: Rotation at which the match was found.
    """

    score: float
    x: int
    y: int
    angle_deg: float


class RefinementJob(NamedTuple):
    """
    One ``(cell, predicted centre, trial angle)`` triple to score at full resolution.

    Positions are centres in unrotated image coordinates; refine maps them onto the
    rotated canvas.
    """

    cell_index: int
    center_x: float
    center_y: float
    angle_deg: float
