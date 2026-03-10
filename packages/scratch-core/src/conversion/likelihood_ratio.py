import pickle
from pathlib import Path
from typing import Self

import numpy as np
from lir.data.models import FeatureData, LLRData
from lir.lrsystems import LRSystem
from pydantic import model_validator

from container_models.base import ConfigBaseModel


class ModelSpecs(ConfigBaseModel):
    """Training data and model types for KM and KNM populations used to calibrate an LR system.

    Holds scores and LLR data for two populations: known matches (KM) and
    known non-matches (KNM), along with the model name used to produce each.

    :param km_model: Identifier of the model used for KM scores.
    :param km_scores: Similarity scores for the KM population.
    :param km_llrs: Log-likelihood ratios for the KM population.
    :param km_llr_intervals: LLR confidence intervals for the KM population, shape (n, 2), or None.
    :param knm_model: Identifier of the model used for KNM scores.
    :param knm_scores: Similarity scores for the KNM population.
    :param knm_llrs: Log-likelihood ratios for the KNM population.
    :param knm_llr_intervals: LLR confidence intervals for the KNM population, shape (n, 2), or None.
    """

    km_model: str
    km_scores: np.ndarray
    km_llrs: np.ndarray
    km_llr_intervals: np.ndarray | None
    knm_model: str
    knm_scores: np.ndarray
    knm_llrs: np.ndarray
    knm_llr_intervals: np.ndarray | None

    @model_validator(mode="after")
    def _validate_matching_lengths(self) -> Self:
        if len(self.km_scores) != len(self.km_llrs):
            raise ValueError("km_scores and km_lrs must have the same length")
        if len(self.knm_scores) != len(self.knm_llrs):
            raise ValueError("knm_scores and knm_lrs must have the same length")
        return self

    @property
    def scores(self) -> np.ndarray:
        """Concatenated KM and KNM similarity scores."""
        return np.concatenate([self.km_scores, self.knm_scores])

    @property
    def llrs(self) -> np.ndarray:
        """Concatenated KM and KNM log-likelihood ratios."""
        return np.concatenate([self.km_llrs, self.knm_llrs])

    @property
    def llr_intervals(self) -> np.ndarray:
        """Concatenated KM and KNM LLR intervals, shape (n, 2)."""
        if self.km_llr_intervals is None or self.knm_llr_intervals is None:
            raise ValueError("Only models with llr_intervals can be used")
        return np.concatenate([self.km_llr_intervals, self.knm_llr_intervals], axis=0)

    @property
    def labels(self) -> np.ndarray:
        """Boolean labels: True for KM samples, False for KNM samples."""
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
) -> ModelSpecs:  # TODO replace with lr_module_scratch
    """Return hardcoded dummy reference data (KM/KNM scores and LLRs).

    .. note::
        This is a placeholder. The ``lr_system_path`` argument is accepted for
        API compatibility but is not used; real reference data will be derived
        from the LR system once ``lr_module_scratch`` is integrated.
    """
    _ = get_lr_system(lr_system_path)
    return ModelSpecs(
        km_model="random",
        km_scores=np.array([0.9, 0.85, 0.78]),
        km_llrs=np.array([2.1, 1.8, 1.5]),
        km_llr_intervals=np.array([[1.9, 2.3], [1.6, 2.0], [1.3, 1.7]]),
        knm_model="random",
        knm_scores=np.array([0.3, 0.25, 0.15, 0.1]),
        knm_llrs=np.array([-1.2, -0.9, -1.5, -2.0]),
        knm_llr_intervals=np.array(
            [[-1.4, -1.0], [-1.1, -0.7], [-1.7, -1.3], [-2.2, -1.8]]
        ),
    )


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
    result = lr_system.apply(FeatureData(features=np.array([[score, n_cells]])))
    return result
