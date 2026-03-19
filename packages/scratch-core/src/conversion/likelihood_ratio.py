from pathlib import Path
from typing import Self
from lrmodule import get_lr_system, get_reference_data

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


def get_reference_data_from_path(
    lr_system_path: Path,
) -> ModelSpecs:
    """Return the reference data of a specific LR system."""
    lr_system = get_lr_system(lr_system_path)
    reference_data = get_reference_data(lr_system_path)
    if reference_data.labels is None:
        raise ValueError("reference data must have labels")
    if not set(reference_data.labels).issubset({0, 1}):
        raise ValueError(
            f"reference data labels must be 0 or 1, got {set(reference_data.labels)}"
        )

    mask = reference_data.labels == 1
    km_scores = FeatureData(
        features=reference_data.features[mask],
        labels=reference_data.labels[mask],
        source_ids=reference_data.source_ids[mask]
        if reference_data.source_ids is not None
        else None,
    )
    km_llr_data = lr_system.apply(km_scores)

    mask = reference_data.labels == 0
    knm_scores = FeatureData(
        features=reference_data.features[mask],
        labels=reference_data.labels[mask],
        source_ids=reference_data.source_ids[mask]
        if reference_data.source_ids is not None
        else None,
    )
    knm_llr_data = lr_system.apply(knm_scores)
    return ModelSpecs(
        km_scores=km_scores.features,
        km_llrs=km_llr_data.llrs,
        km_llr_intervals=km_llr_data.llr_intervals,
        knm_scores=knm_scores.features,
        knm_llrs=knm_llr_data.llrs,
        knm_llr_intervals=knm_llr_data.llr_intervals,
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
    result = lr_system.apply(FeatureData(features=np.array([[0, score, n_cells]])))
    return result


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
