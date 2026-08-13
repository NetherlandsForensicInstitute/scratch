from pathlib import Path

from lir import FeatureData
from lrmodule import get_lr_system, get_reference_data

from conversion.likelihood_ratio.data_formats import ModelSpecs


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
