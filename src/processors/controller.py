from http import HTTPStatus
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from conversion.data_formats import Mark
from conversion.plots.data_formats import LlrTransformationData
from conversion.plots.plot_score_llr_transformation import plot_score_llr_transformation
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.plots.utils import figure_to_array
from conversion.profile_correlator import MarkCorrelationResult, Profile, correlate_striation_marks
from fastapi import HTTPException
from lir.data.models import FeatureData
from lir.lrsystems.lrsystems import LRSystem
from loguru import logger
from PIL import Image


def compare_striation_marks(
    mark_ref: Mark, mark_comp: Mark, profile_ref: Profile, profile_comp: Profile
) -> MarkCorrelationResult:
    """Calculate correlation between two striation marks."""
    mark_correlations = correlate_striation_marks(
        mark_reference=mark_ref,
        mark_compared=mark_comp,
        profile_reference=profile_ref,
        profile_compared=profile_comp,
    )
    if not mark_correlations:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "No correlations were found.")
    logger.debug("correlations are calculated")
    return mark_correlations


def save_striation_comparison_plots(  # noqa: PLR0913
    mark_ref: Mark,
    mark_comp: Mark,
    mark_correlations: MarkCorrelationResult,
    files_to_save: dict[str, Path],
    meta_data_compare: dict[str, str],
    meta_data_ref: dict[str, str],
) -> None:
    """Create and save the plots of the processed markings."""
    plots = plot_striation_comparison_results(
        mark_reference=mark_ref,
        mark_compared=mark_comp,
        mark_reference_aligned=mark_correlations.mark_reference_aligned,
        mark_compared_aligned=mark_correlations.mark_compared_aligned,
        profile_reference_aligned=mark_correlations.profile_reference_aligned,
        profile_compared_aligned=mark_correlations.profile_compared_aligned,
        metrics=mark_correlations.comparison_results,
        metadata_reference=meta_data_ref,
        metadata_compared=meta_data_compare,
    )
    logger.debug("striation comparison plots generated")
    # TODO: update this dict to a Pydantic class.
    #  so plots.attribute is linked to get_file_path(ComparisonResponse.attribute)
    Image.fromarray(plots.similarity_plot).save(files_to_save["similarity_plot"])
    Image.fromarray(plots.side_by_side_heatmap).save(files_to_save["mark1_vs_moved_mark2"])
    Image.fromarray(plots.comparison_overview).save(files_to_save["comparison_overview"])
    Image.fromarray(plots.filtered_compared_heatmap).save(files_to_save["mark_comp_filtered_surfacemap"])
    Image.fromarray(plots.filtered_reference_heatmap).save(files_to_save["mark_ref_filtered_surfacemap"])


def save_lr_overview_plot(
    system: LRSystem,
    score: float,
    lr: float,
    score_max: float,
    output_path: Path,
) -> None:
    """Generate and save the LLR transformation overview plot."""
    scores = np.linspace(0, score_max, 100)
    llr_result = system.apply(FeatureData(features=scores.reshape(-1, 1)))
    llrs_at5 = llr_result.llr_intervals[:, 0] if llr_result.has_intervals else llr_result.llrs
    llrs_at95 = llr_result.llr_intervals[:, 1] if llr_result.has_intervals else llr_result.llrs
    llr_data = LlrTransformationData(
        scores=scores,
        llrs=llr_result.llrs,
        llrs_at5=llrs_at5,
        llrs_at95=llrs_at95,
        score_llr_point=(float(score), lr),
    )
    fig, ax = plt.subplots()
    plot_score_llr_transformation(ax, llr_data)
    Image.fromarray(figure_to_array(fig)).save(output_path)
    plt.close(fig)
