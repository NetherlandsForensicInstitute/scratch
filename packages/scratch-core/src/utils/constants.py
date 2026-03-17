from enum import IntEnum


class RegressionOrder(IntEnum):
    GAUSSIAN_WEIGHTED_AVERAGE = 0
    LOCAL_PLANAR = 1
    LOCAL_QUADRATIC = 2
