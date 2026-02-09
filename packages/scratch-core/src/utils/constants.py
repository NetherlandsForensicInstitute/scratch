from enum import IntEnum, auto


class RegressionOrder(IntEnum):
    GAUSSIAN_WEIGHTED_AVERAGE = auto()
    LOCAL_PLANAR = auto()
    LOCAL_QUADRATIC = auto()
