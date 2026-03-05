from http import HTTPStatus
from pathlib import Path

from conversion.data_formats import Mark, ReferenceData
from conversion.plots.data_formats import HistogramData, ImpressionComparisonMetrics, LlrTransformationData
from conversion.plots.plot_ccf_comparison_overview import plot_ccf_comparison_overview
from conversion.plots.plot_cmc_comparison_overview import plot_cmc_comparison_overview
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.profile_correlator import MarkCorrelationResult, Profile, correlate_striation_marks
from fastapi import HTTPException
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
    metadata_compared: dict[str, str],
    metadata_reference: dict[str, str],
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
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
    )
    logger.debug("striation comparison plots generated")
    # TODO: update this dict to a Pydantic class.
    #  so plots.attribute is linked to get_file_path(ComparisonResponse.attribute)
    Image.fromarray(plots.similarity_plot).save(files_to_save["similarity_plot"])
    Image.fromarray(plots.side_by_side_heatmap).save(files_to_save["side_by_side_heatmap"])
    Image.fromarray(plots.comparison_overview).save(files_to_save["comparison_overview"])
    Image.fromarray(plots.filtered_compared_heatmap).save(files_to_save["filtered_compared_heatmap"])
    Image.fromarray(plots.filtered_reference_heatmap).save(files_to_save["filtered_reference_heatmap"])


def save_lr_impression_plot(  # noqa: PLR0913
    reference_data: ReferenceData,
    mark_ref: Mark,
    mark_comp: Mark,
    metrics: ImpressionComparisonMetrics,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    results_metadata: dict[str, str],
    score: float,
    lr: float,
    output_path: Path,
):
    """
    Generate and save a CMC comparison overview plot for impression marks.

    Combines surface visualizations, cell correlation metrics, score histograms,
    and LLR transformation curves into a single overview image.

    :param reference_data: Reference population data with KM/KNM scores and LLRs.
    :param mark_ref: Filtered reference mark surface.
    :param mark_comp: Filtered compared mark surface.
    :param metrics: Cell and area correlation metrics from the CMC comparison.
    :param metadata_reference: Display metadata for the reference mark.
    :param metadata_compared: Display metadata for the compared mark.
    :param results_metadata: Formatted summary of comparison results for display.
    :param score: CMC score for the current case comparison.
    :param lr: Log-likelihood ratio for the current case comparison.
    :param output_path: Path to save the output PNG image.
    """
    plot = plot_cmc_comparison_overview(
        mark_reference_filtered=mark_ref,
        mark_compared_filtered=mark_comp,
        metrics=metrics,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
        results_metadata=results_metadata,
        histogram_data=HistogramData(scores=reference_data.scores, labels=reference_data.labels, new_score=score),
        llr_data=LlrTransformationData(
            scores=reference_data.scores,
            llrs=reference_data.llrs,
            llrs_at5=reference_data.llrs_at5,
            llrs_at95=reference_data.llrs_at95,
            score_llr_point=(score, lr),
        ),
    )
    Image.fromarray(plot).save(output_path)


def save_lr_striation_plot(  # noqa: PLR0913
    reference_data: ReferenceData,
    mark_ref: Mark,
    mark_comp: Mark,
    mark_ref_aligned: Mark,
    mark_comp_aligned: Mark,
    metadata_reference: dict[str, str],
    metadata_compared: dict[str, str],
    results_metadata: dict[str, str],
    score: float,
    lr: float,
    output_path: Path,
):
    """
    Generate and save a CCF comparison overview plot for striation marks.

    Combines surface visualizations of both full and aligned marks, score
    histograms, and LLR transformation curves into a single overview image.

    :param reference_data: Reference population data with KM/KNM scores and LLRs.
    :param mark_ref: Filtered reference mark surface.
    :param mark_comp: Filtered compared mark surface.
    :param mark_ref_aligned: Reference mark trimmed to the overlap region.
    :param mark_comp_aligned: Compared mark trimmed to the overlap region.
    :param metadata_reference: Display metadata for the reference mark.
    :param metadata_compared: Display metadata for the compared mark.
    :param results_metadata: Formatted summary of comparison results for display.
    :param score: CCF score for the current case comparison.
    :param lr: Log-likelihood ratio for the current case comparison.
    :param output_path: Path to save the output PNG image.
    """
    plot = plot_ccf_comparison_overview(
        mark_reference_filtered=mark_ref,
        mark_compared_filtered=mark_comp,
        mark_reference_aligned=mark_ref_aligned,
        mark_compared_aligned=mark_comp_aligned,
        metadata_reference=metadata_reference,
        metadata_compared=metadata_compared,
        results_metadata=results_metadata,
        histogram_data=HistogramData(scores=reference_data.scores, labels=reference_data.labels, new_score=score),
        histogram_data_transformed=HistogramData(
            scores=reference_data.scores, labels=reference_data.labels, new_score=score
        ),
        llr_data=LlrTransformationData(
            scores=reference_data.scores,
            llrs=reference_data.llrs,
            llrs_at5=reference_data.llrs_at5,
            llrs_at95=reference_data.llrs_at95,
            score_llr_point=(score, lr),
        ),
    )
    Image.fromarray(plot).save(output_path)
