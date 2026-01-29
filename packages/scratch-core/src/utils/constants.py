from enum import Enum


class RegressionOrder(Enum):
    GAUSSIAN_WEIGHTED_AVERAGE = 0
    LOCAL_PLANAR = 1
    LOCAL_QUADRATIC = 2
