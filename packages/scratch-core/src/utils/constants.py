from enum import Enum, auto


class RegressionOrder(Enum):
    GAUSSIAN_WEIGHTED_AVERAGE = auto()
    LOCAL_PLANAR = auto()
    LOCAL_QUADRATIC = auto()
