from http import HTTPStatus
from pathlib import Path

from conversion.data_formats import Mark, MarkMetadata
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.profile_correlator import MarkCorrelationResult, Profile, correlate_striation_marks
from fastapi import HTTPException
from loguru import logger
from PIL import Image

from extractors.constants import ComparisonStriationFiles


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
        logger.error("profiles could not be aligned: insufficient overlap between marks")
        raise HTTPException(
            HTTPStatus.UNPROCESSABLE_ENTITY, "profiles could not be aligned: insufficient overlap between marks"
        )
    logger.debug("correlations are calculated")
    return mark_correlations


def save_striation_comparison_plots(  # noqa: PLR0913
    mark_ref: Mark,
    mark_comp: Mark,
    mark_correlations: MarkCorrelationResult,
    working_dir: Path,
    files_to_save: type[ComparisonStriationFiles],
    meta_data_compare: MarkMetadata,
    meta_data_ref: MarkMetadata,
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
    Image.fromarray(plots.similarity_plot).save(files_to_save.similarity_plot.get_file_path(working_dir))
    Image.fromarray(plots.side_by_side_heatmap).save(files_to_save.side_by_side_heatmap.get_file_path(working_dir))
    Image.fromarray(plots.comparison_overview).save(files_to_save.comparison_overview.get_file_path(working_dir))
    Image.fromarray(plots.filtered_compared_heatmap).save(
        files_to_save.filtered_compared_heatmap.get_file_path(working_dir)
    )
    Image.fromarray(plots.filtered_reference_heatmap).save(
        files_to_save.filtered_reference_heatmap.get_file_path(working_dir)
    )
