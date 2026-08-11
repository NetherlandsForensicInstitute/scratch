from typing import NamedTuple


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

    Positions are centres in unrotated image coordinates; :mod:`.refine` maps them onto the
    rotated canvas.
    """

    cell_index: int
    center_x: float
    center_y: float
    angle_deg: float
