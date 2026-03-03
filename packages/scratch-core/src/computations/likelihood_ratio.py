import pickle
from pathlib import Path

import numpy as np
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem


def get_lr_system(lr_system_path: Path) -> LRSystem:
    """Load an LR system from a pickle file."""
    with lr_system_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def calculate_lr(
    score: float, lr_system: LRSystem, n_cells: int | None = None
) -> float:
    """Calculate likelihood ratio by applying the LR system to the score.

    For striation marks, pass only ``score`` (a correlation coefficient).
    For impression marks, pass both ``score`` (CMC count) and ``n_cells``
    (total cells analysed); both are forwarded as features to the LR system.
    """
    features = [score, n_cells] if n_cells is not None else [score]
    result = lr_system.apply(FeatureData(features=np.array([features])))
    return float(result.llrs[0])
