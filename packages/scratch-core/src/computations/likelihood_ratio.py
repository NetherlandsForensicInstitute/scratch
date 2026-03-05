import pickle
from pathlib import Path

import numpy as np
from lir.data.models import FeatureData, LLRData
from lir.lrsystems import LRSystem

from conversion.data_formats import ReferenceData


def get_lr_system(
    lr_system_path: Path,
) -> LRSystem:  # TODO replace with lr_module_scratch
    """Load an LR system from a pickle file."""
    with lr_system_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def get_reference_data(
    lr_system_path: Path,
) -> ReferenceData:  # TODO replace with lr_module_scratch
    """Load an LR system from a pickle file."""
    _ = get_lr_system(lr_system_path)
    return ReferenceData(
        km_model="random",
        km_scores=np.array([0.9, 0.85, 0.78]),
        km_llr_data=LLRData(
            features=np.array(
                [
                    [2.1, 1.9, 2.3],
                    [1.8, 1.6, 2.0],
                    [1.5, 1.3, 1.7],
                ]
            )
        ),
        knm_model="random",
        knm_scores=np.array([0.3, 0.25, 0.15, 0.1]),
        knm_llr_data=LLRData(
            features=np.array(
                [
                    [-1.2, -1.4, -1.0],
                    [-0.9, -1.1, -0.7],
                    [-1.5, -1.7, -1.3],
                    [-2.0, -2.2, -1.8],
                ]
            )
        ),
    )
    # noqa: S301


def calculate_lr_striation(lr_system: LRSystem, score: float) -> LLRData:
    """
    Calculate likelihood ratio for striation marks.

    :param lr_system: Trained LR system to apply.
    :param score: Correlation coefficient between two striation profiles.
    """
    result = lr_system.apply(FeatureData(features=np.array([[score]])))
    return result


def calculate_lr_impression(lr_system: LRSystem, score: int, n_cells: int) -> LLRData:
    """
    Calculate likelihood ratio for impression marks.

    :param lr_system: Trained LR system to apply.
    :param score: CMC count (number of matching cells).
    :param n_cells: Total number of cells analyzed.
    """
    result = lr_system.apply(FeatureData(features=np.array([[score, n_cells]])))
    return result
