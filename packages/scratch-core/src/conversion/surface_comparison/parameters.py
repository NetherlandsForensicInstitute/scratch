from dataclasses import dataclass, field
import numpy as np


@dataclass
class ComparisonParams:
    """
    Settings for the Congruent Matching Cells (CMC) algorithm logic.

    :param cell_size: nominal size [width, height] in micrometers, shape (2,).
    :param minimum_fill_fraction: minimum valid data required to process a cell.
    :param correlation_threshold: minimum ACCF for a cell to be considered.
    :param angle_threshold: maximum allowed angular deviation (degrees).
    :param position_threshold: maximum allowed positional deviation (micrometers).
    :param search_angle_min: minimum angle to test during rotation search (degrees).
    :param search_angle_max: maximum angle to test during rotation search (degrees).
    :param search_angle_step: increment for the rotation search (degrees).
    """

    cell_size: np.ndarray = field(default_factory=lambda: np.array([1000.0, 1000.0]))
    minimum_fill_fraction: float = 0.5
    correlation_threshold: float = 0.4
    angle_threshold: float = 2.0
    position_threshold: float = 100.0
    search_angle_min: float = -5.0
    search_angle_max: float = 5.0
    search_angle_step: float = 0.5
