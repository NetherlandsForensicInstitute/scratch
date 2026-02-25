from http import HTTPStatus
from pathlib import Path
from typing import Type

from conversion.data_formats import Mark
from conversion.export.mark import load_mark_from_path
from conversion.export.profile import load_profile_from_path
from conversion.plots.data_formats import StriationComparisonPlots
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.profile_correlator import MarkCorrelationResult, correlate_striation_marks
from fastapi import HTTPException
from loguru import logger
from PIL import Image

from extractors.schemas import ComparisonResponse


def calculate_striation_plots(
    mark_ref: Mark,
    mark_comp: Mark,
    compare_path: Path,
    ref_path: Path,
) -> MarkCorrelationResult:
    """calculate striation plots."""
    mark_ref_profile = load_profile_from_path(path=ref_path, stem="profile")
    mark_comp_profile = load_profile_from_path(path=compare_path, stem="profile")
    logger.debug("Profile loaded")

    mark_correlations = correlate_striation_marks(
        mark_reference=mark_ref,
        mark_compared=mark_comp,
        profile_reference=mark_ref_profile,
        profile_compared=mark_comp_profile,
    )
    if not mark_correlations:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "we have found no correlations, I know shouldn't happens")
    logger.debug("correlations made")
    return mark_correlations


def save_plots(
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
    logger.debug("plots made")
    # TODO: update these dict to a Pydantic class. so plots.attribute is linked to get_file_path(ComparisonResponse.attribute)
    Image.fromarray(plots.similarity_plot).save(files_to_save["similarity_plot"])
    Image.fromarray(plots.side_by_side_heatmap).save(files_to_save["mark1_vs_moved_mark2"])
    Image.fromarray(plots.comparison_overview).save(files_to_save["comparison_overview"])
    Image.fromarray(plots.filtered_compared_heatmap).save(files_to_save["mark_comp_filtered_surfacemap"])
    Image.fromarray(plots.filtered_reference_heatmap).save(files_to_save["mark_ref_filtered_surfacemap"])
