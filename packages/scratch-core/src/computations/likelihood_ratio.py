import pickle
from pathlib import Path

import numpy as np
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem


def get_lr_system(lr_system_path: Path) -> LRSystem:
    """Load an LR system from a pickle file."""
    with lr_system_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def calculate_lr_striation(lr_system: LRSystem, score: float) -> float:
    """Calculate likelihood ratio for striation marks.

    Args:
        score: Correlation coefficient between two striation profiles.
        lr_system: Trained LR system to apply.
    """
    result = lr_system.apply(FeatureData(features=np.array([[score]])))
    return float(result.llrs[0])


def calculate_lr_impression(lr_system: LRSystem, score: int, n_cells: int) -> float:
    """Calculate likelihood ratio for impression marks.

    Args:
        score: CMC count (number of matching cells).
        n_cells: Total number of cells analysed.
        lr_system: Trained LR system to apply.
    """
    result = lr_system.apply(FeatureData(features=np.array([[score, n_cells]])))
    return float(result.llrs[0])
