import pickle
from pathlib import Path
from typing import Self

import numpy as np
from lir.data.models import FeatureData, LLRData
from lir.lrsystems import LRSystem
from pydantic import model_validator

from container_models.base import ConfigBaseModel


class ReferenceData(ConfigBaseModel):
    km_model: str
    km_scores: np.ndarray
    km_llr_data: LLRData
    knm_model: str
    knm_scores: np.ndarray
    knm_llr_data: LLRData

    @model_validator(mode="after")
    def _validate_matching_lengths(self) -> Self:
        if len(self.km_scores) != len(self.km_llr_data.llrs):
            raise ValueError("km_scores and km_lrs must have the same length")
        if len(self.knm_scores) != len(self.knm_llr_data.llrs):
            raise ValueError("knm_scores and knm_lrs must have the same length")
        return self

    @property
    def scores(self) -> np.ndarray:
        return np.concatenate([self.km_scores, self.knm_scores])

    @property
    def llrs(self) -> np.ndarray:
        return np.concatenate([self.km_llr_data.llrs, self.knm_llr_data.llrs])

    @property
    def llr_intervals(self) -> np.ndarray:
        """Concatenated KM and KNM LLR intervals, shape (n, 2)."""
        km = self.km_llr_data.llr_intervals
        knm = self.knm_llr_data.llr_intervals
        if km is None or knm is None:
            raise ValueError("Only models with llr_intervals can be used")
        return np.concatenate([km, knm], axis=0)

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            [
                np.ones(len(self.km_scores), dtype=bool),
                np.zeros(len(self.knm_scores), dtype=bool),
            ]
        )


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


def transform_ccf_scores(scores: np.ndarray) -> np.ndarray:
    """
    Transform CCF scores from [-1, +1] to [-inf, +inf] using a log10 logit.

    Rescales to [0, 1] then applies log-odds (base 10):
        y = (score + 1) / 2
        transformed = log10(y / (1 - y))

    Boundary values are clipped by one ULP to avoid infinite results.

    :param scores: 1-D array of raw CCF scores in [-1, +1].
    :returns: 1-D array of transformed scores.
    """
    eps = np.finfo(float).eps
    clipped = np.clip(scores, -1 + eps, 1 - eps)
    y = (clipped + 1) / 2
    return np.log10(y / (1 - y))
