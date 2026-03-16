from dataclasses import asdict

from conversion.export.mark import load_mark_from_path, save_mark
from conversion.export.profile import load_profile_from_path
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

from constants import LIGHT_SOURCES, OBSERVER, ProcessorEndpoint, RoutePrefix
from extractors.constants import ComparisonImpressionFiles, ComparisonStriationFiles, LRFiles
from extractors.schemas import (
    ComparisonResponseImpression,
    ComparisonResponseStriation,
    ComparisonResponseStriationURL,
    LRResponse,
    LRResponseURL,
    LRStriationResponse,
)
from file_services import create_vault
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from processors.controller import (
    compare_striation_marks,
    process_lr_impression,
    process_lr_striation,
    save_striation_comparison_plots,
)
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScore,
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
async def calculate_score_impression(impression: CalculateScore) -> ComparisonResponseImpression:
    """Compare two impression profiles."""
    vault = create_vault(impression.tag)
    return ComparisonResponseImpression.from_enum(enum=ComparisonImpressionFiles, base_url=vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_SCORE_STRIATION}",
    summary="Compare two striation profiles",
    description="""
    Reads preprocessed striation profiles from the comparison and reference directories,
    performs pairwise comparison, and calculates a score (CMC).
    The score, together with plots, are saved and made available via URLs.
    """,
    responses={
        422: {"description": "Profiles could not be aligned due to insufficient overlap"},
    },
)
async def calculate_score_striation(striation_params: CalculateScore) -> ComparisonResponseStriation:
    """Compare two striation profiles."""
    logger.debug("starting calculate score striation")
    vault = create_vault(striation_params.tag)
    mark_ref = load_mark_from_path(path=striation_params.mark_dir_ref, stem="processed")
    mark_comp = load_mark_from_path(path=striation_params.mark_dir_comp, stem="processed")
    profile_ref = load_profile_from_path(path=striation_params.mark_dir_ref, stem="profile")
    profile_comp = load_profile_from_path(path=striation_params.mark_dir_comp, stem="profile")
    logger.debug("marks & profiles loaded")
    comparison_result = compare_striation_marks(
        mark_ref=mark_ref, mark_comp=mark_comp, profile_ref=profile_ref, profile_comp=profile_comp
    )
    save_striation_comparison_plots(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_correlations=comparison_result,
        working_dir=vault.resource_path,
        files_to_save=ComparisonStriationFiles,
        metadata_reference=striation_params.param.metadata_reference,
        metadata_compared=striation_params.param.metadata_compared,
    )
    save_mark(
        comparison_result.mark_reference_aligned,
        path=ComparisonStriationFiles.mark_reference_aligned_data.get_file_path(vault.resource_path),
    )
    save_mark(
        comparison_result.mark_compared_aligned,
        path=ComparisonStriationFiles.mark_compared_aligned_data.get_file_path(vault.resource_path),
    )
    surface_map_pipeline(
        comparison_result.mark_reference_aligned.scan_image,
        ComparisonStriationFiles.mark_ref_surfacemap.get_file_path(vault.resource_path),
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(
        comparison_result.mark_reference_aligned.scan_image,
        ComparisonStriationFiles.mark_ref_preview.get_file_path(vault.resource_path),
    )
    surface_map_pipeline(
        comparison_result.mark_compared_aligned.scan_image,
        ComparisonStriationFiles.mark_comp_surfacemap.get_file_path(vault.resource_path),
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(
        comparison_result.mark_compared_aligned.scan_image,
        ComparisonStriationFiles.mark_comp_preview.get_file_path(vault.resource_path),
    )
    logger.debug(f"images saved in:{vault.resource_path}")

    return ComparisonResponseStriation(
        urls=ComparisonResponseStriationURL.from_enum(enum=ComparisonStriationFiles, base_url=vault.access_url),
        comparison_results=asdict(comparison_result.comparison_results),
    )


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}",
    summary="Calculate likelihood ratio for impression mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of impression marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_lr_impression(lr_input: CalculateLRImpression) -> LRResponse:
    """Calculate likelihood ratio for impression mark comparison."""
    vault = create_vault(lr_input.tag)
    result = process_lr_impression(lr_input=lr_input, working_dir=vault.resource_path)
    return LRResponse(
        urls=LRResponseURL.from_enum(enum=LRFiles, base_url=vault.access_url),
        lr=result.log_lr,
        km_scores=result.km_scores,
        knm_scores=result.knm_scores,
        km_llr=result.km_llr,
        knm_llr=result.knm_llr,
        km_llr_lower_ci=result.km_llr_lower_ci,
        km_llr_upper_ci=result.km_llr_upper_ci,
        knm_llr_lower_ci=result.knm_llr_lower_ci,
        knm_llr_upper_ci=result.knm_llr_upper_ci,
    )


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_STRIATION}",
    summary="Calculate likelihood ratio for striation mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of striation marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_lr_striation(lr_input: CalculateLRStriation) -> LRStriationResponse:
    """Calculate likelihood ratio for striation mark comparison."""
    vault = create_vault(lr_input.tag)
    result = process_lr_striation(lr_input=lr_input, working_dir=vault.resource_path)
    return LRStriationResponse(
        urls=LRResponseURL.from_enum(enum=LRFiles, base_url=vault.access_url),
        lr=result.log_lr,
        km_scores=result.km_scores,
        knm_scores=result.knm_scores,
        km_llr=result.km_llr,
        knm_llr=result.knm_llr,
        km_llr_lower_ci=result.km_llr_lower_ci,
        km_llr_upper_ci=result.km_llr_upper_ci,
        knm_llr_lower_ci=result.knm_llr_lower_ci,
        knm_llr_upper_ci=result.knm_llr_upper_ci,
        km_scores_transformed=result.km_scores_transformed,
        knm_scores_transformed=result.knm_scores_transformed,
    )
