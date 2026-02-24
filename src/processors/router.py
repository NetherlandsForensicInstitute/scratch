from dataclasses import asdict
from http import HTTPStatus

from conversion.export.mark import load_mark_from_path
from conversion.export.profile import load_profile_from_path
from conversion.plots.plot_striation import plot_striation_comparison_results
from conversion.profile_correlator import correlate_profiles, correlate_striation_marks
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from loguru import logger
from PIL import Image

from constants import LIGHT_SOURCES, OBSERVER, ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation, LRResponse, LRResponseURL
from file_services import create_vault
from models import DirectoryAccess
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScoreImpression,
    CalculateScoreStriation,
)

processors = APIRouter(
    prefix=f"/{RoutePrefix.PROCESSOR}",
    tags=[RoutePrefix.PROCESSOR],
)


@processors.get(
    path=ProcessorEndpoint.ROOT,
    summary="Redirect to processor documentation",
    description="""Redirects to the processor section in the API documentation.""",
    include_in_schema=False,
)
async def processor_root() -> RedirectResponse:
    """
    Redirect to the processor section in Swagger docs.

    This endpoint redirects users to the processor tag section in the
    interactive API documentation at /docs.

    :return: RedirectResponse to the processor documentation section.
    """
    return RedirectResponse(url=f"/docs#operations-tag-{RoutePrefix.PROCESSOR}")


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_SCORE_IMPRESSION}",
    summary="Compare two impression marks.",
    description="""
    Reads preprocessed impression marks from the comparison and reference directories,
    performs pairwise comparison, and calculates a score (correlation coefficient).
    The score, together with plots, are saved and made available via URLs.
    """,
    include_in_schema=False,
)
async def calculate_score_impression(impression: CalculateScoreImpression) -> ComparisonResponseImpression:
    """Compare two impression profiles."""
    vault = DirectoryAccess()  # type: ignore
    return ComparisonResponseImpression.generate_urls(vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_SCORE_STRIATION}",
    summary="Compare two striation profiles",
    description="""
    Reads preprocessed striation profiles from the comparison and reference directories,
    performs pairwise comparison, and calculates a score (CMC).
    The score, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_score_striation(striation_params: CalculateScoreStriation) -> ComparisonResponseStriation:
    """Compare two striation profiles."""
    logger.debug("starting calculate score striation")
    vault = create_vault(striation_params.tag)
    vault.resource_path.exists()
    logger.debug(f"working_dir made in:{vault.resource_path}")
    mark_ref = load_mark_from_path(path=striation_params.mark_ref, stem="processed")
    mark_ref_profile = load_profile_from_path(path=striation_params.mark_ref, stem="profile")
    logger.debug("reference striation loaded")
    mark_comp = load_mark_from_path(path=striation_params.mark_comp, stem="processed")
    mark_comp_profile = load_profile_from_path(path=striation_params.mark_comp, stem="profile")
    logger.debug("compute striation loaded")

    mark_correlations = correlate_striation_marks(
        mark_reference=mark_ref,
        mark_compared=mark_comp,
        profile_reference=mark_ref_profile,
        profile_compared=mark_comp_profile,
    )
    if not mark_correlations:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "we have found no correlations, I know shouldn't happens")
    logger.debug("correlations made")
    plots = plot_striation_comparison_results(
        mark_reference=mark_ref,
        mark_compared=mark_comp,
        mark_reference_aligned=mark_correlations.mark_reference_aligned,
        mark_compared_aligned=mark_correlations.mark_compared_aligned,
        profile_reference_aligned=mark_correlations.profile_reference_aligned,
        profile_compared_aligned=mark_correlations.profile_compared_aligned,
        metrics=mark_correlations.comparison_results,
        metadata_reference=striation_params.param.metadata_reference,
        metadata_compared=striation_params.param.metadata_compared,
    )
    logger.debug("plots made")

    expected_files = ComparisonResponseStriation.get_files(vault.resource_path)

    Image.fromarray(plots.similarity_plot).save(expected_files["similarity_plot"])

    Image.fromarray(plots.side_by_side_heatmap).save(expected_files["mark1_vs_moved_mark2"])
    Image.fromarray(plots.comparison_overview).save(expected_files["comparison_overview"])

    Image.fromarray(plots.filtered_compared_heatmap).save(expected_files["mark_comp_filtered_surfacemap"])
    Image.fromarray(plots.filtered_reference_heatmap).save(expected_files["mark_ref_filtered_surfacemap"])

    surface_map_pipeline(mark_ref.scan_image, expected_files["mark_ref_surfacemap"], LIGHT_SOURCES, OBSERVER)
    preview_pipeline(mark_ref.scan_image, expected_files["mark_ref_depthmap"])
    surface_map_pipeline(mark_comp.scan_image, expected_files["mark_comp_surfacemap"], LIGHT_SOURCES, OBSERVER)
    preview_pipeline(mark_comp.scan_image, expected_files["mark_comp_depthmap"])
    logger.debug(f"images saved in:{vault.resource_path}")

    return ComparisonResponseStriation.generate_urls(vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}",
    summary="Calculate likelihood ratio for impression mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of impression marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_lr_impression(impression: CalculateLRImpression) -> LRResponse:
    """Calculate likelihood ratio for impression mark comparison."""
    vault = DirectoryAccess()  # type: ignore
    # TODO::
    # - create controllers module
    # - This section below need to be moved to controllers.py
    #
    # controllers.py
    # def compute_n_plot_lr(ref: Mark, comp: Mark, score: int: lr_system: Path) -> float:
    #     system=get_lr_system(lr_system)
    #     lr = calculate_lr(
    #       score,
    #       striation.n_cells,
    #       use_intervals=bool,
    #       lr_system=system,
    #     )
    #     plot_lr_result(system, *read_mark_file(mark_ref, mark_comp), striation.score)
    #     return lr
    #
    # return LRResponse.generate_urls(
    #     vault.access_url,
    #     lr=controllers.comupute_lr(
    #         impression.mark_ref,
    #         impression.mark_comp,
    #         impression.score,
    #         impression.lr_system,
    #     )
    # )
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=42)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_STRIATION}",
    summary="Calculate likelihood ratio for striation mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of striation marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_lr_striation(striation: CalculateLRStriation) -> LRResponse:
    """TODO."""
    vault = DirectoryAccess()  # type: ignore
    # return LRResponse.generate_urls(
    #     vault.access_url,
    #     lr=controllers.compute_n_plot_lr(
    #         striation.mark_ref,
    #         striation.mark_comp,
    #         striation.score,
    #         striation.lr_system,
    #     )
    # )
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=42)
