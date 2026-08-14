import numpy as np
from lir import LLRData, FeatureData
from lir.lrsystems import LRSystem


def calculate_lr_striation(lr_system: LRSystem, score: float) -> LLRData:
    """
    Calculate likelihood ratio for striation marks.

    :param lr_system: Trained LR system to apply.
    :param score: Correlation coefficient between two striation profiles.
    """
    log10_lr_data = lr_system.apply(FeatureData(features=np.array([[score]])))
    return log10_lr_data


def calculate_lr_impression(lr_system: LRSystem, score: int, n_cells: int) -> LLRData:
    """
    Calculate likelihood ratio for impression marks.

    :param lr_system: Trained LR system to apply.
    :param score: CMC count (number of matching cells).
    :param n_cells: Total number of cells analyzed.
    """
    result = lr_system.apply(FeatureData(features=np.array([[0, score, n_cells]])))
    return result
