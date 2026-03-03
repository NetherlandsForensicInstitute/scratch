import pickle
from pathlib import Path

import numpy as np
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem


def get_lr_system(lr_system_path: Path) -> LRSystem:
    """Load an LR system from a pickle file."""
    with lr_system_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def calculate_lr(score: float, lr_system: LRSystem) -> float:
    """Calculate likelihood ratio by applying the LR system to the score."""
    result = lr_system.apply(FeatureData(features=np.array([[score]])))
    return float(result.llrs[0])
