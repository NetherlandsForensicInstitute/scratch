from typing import Self

import numpy as np
from lir.data.models import FeatureData, LLRData, InstanceData
from lir.lrsystems import LRSystem
from pydantic import model_validator

from container_models.base import ConfigBaseModel


class ModelSpecs(ConfigBaseModel):
    """Training data and model types for KM and KNM populations used to calibrate an LR system.

    Holds scores and LLR data for two populations: known matches (KM) and
    known non-matches (KNM), along with the model name used to produce each.

    :param km_scores: Similarity scores for the KM population.
    :param km_llrs: Log-likelihood ratios for the KM population.
    :param km_llr_intervals: LLR confidence intervals for the KM population, shape (n, 2), or None.
    :param knm_scores: Similarity scores for the KNM population.
    :param knm_llrs: Log-likelihood ratios for the KNM population.
    :param knm_llr_intervals: LLR confidence intervals for the KNM population, shape (n, 2), or None.
    """

    km_scores: np.ndarray
    km_llrs: np.ndarray
    km_llr_intervals: np.ndarray | None
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


class DummyLRSystem(LRSystem):  # pragma: no cover
    """Minimal LR system for testing."""

    def apply(self, instances: InstanceData) -> LLRData:
        """Return dummy results."""
        assert isinstance(instances, FeatureData)
        n = len(instances.features)
        # 3 columns: llr, lower_ci, upper_ci
        features = np.column_stack(
            [
                np.zeros(n),  # llrs
                -np.ones(n),  # lower interval
                np.ones(n),  # upper interval
            ]
        )
        return LLRData(features=features)
