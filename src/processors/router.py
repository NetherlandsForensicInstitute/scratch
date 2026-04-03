from dataclasses import asdict
from http import HTTPStatus

from conversion.export.mark import load_mark_from_path, save_mark
from conversion.export.profile import load_profile_from_path
from conversion.surface_comparison.models import ProcessedMark
from conversion.surface_comparison.pipeline import compare_surfaces
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

from constants import LIGHT_SOURCES, OBSERVER, ProcessorEndpoint, RoutePrefix
from file_services import create_vault
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from processors.constants import ComparisonImpressionFiles, ComparisonStriationFiles, LRFiles
from processors.controller import (
    compare_striation_marks,
    process_lr_impression,
    process_lr_striation,
    save_impression_comparison_plots,
    save_striation_comparison_plots,
)
from processors.schemas import (
    CalculateLRImpression,
    CalculateLRStriation,
    CalculateScore,
    CalculateScoreImpression,
    ComparisonResponseImpression,
    ComparisonResponseImpressionURL,
    ComparisonResponseStriation,
    ComparisonResponseStriationURL,
    LRResponse,
    LRResponseURL,
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
    performs pairwise CMC (Congruent Matching Cells) comparison, and calculates a score.
    The score, together with plots, are saved and made available via URLs.
    """,
    include_in_schema=False,
    responses={
        HTTPStatus.NOT_FOUND: {"description": "mark file not found"},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "invalid mark data or comparison failed"},
    },
)
async def calculate_score_impression(impression_params: CalculateScoreImpression) -> ComparisonResponseImpression:
    """Compare two impression profiles."""
    logger.debug("starting calculate score impression")
    vault = create_vault(impression_params.tag)

    mark_ref = load_mark_from_path(path=impression_params.mark_dir_ref, stem="processed")
    mark_ref_raw = load_mark_from_path(path=impression_params.mark_dir_ref, stem="mark")
    mark_ref_processed = ProcessedMark(mark_ref, mark_ref_raw)
    mark_comp = load_mark_from_path(path=impression_params.mark_dir_comp, stem="processed")
    mark_comp_raw = load_mark_from_path(path=impression_params.mark_dir_comp, stem="mark")
    mark_comp_processed = ProcessedMark(mark_comp, mark_comp_raw)
    logger.debug("marks loaded")

    cmc_result = compare_surfaces(
        reference_mark=mark_ref_processed,
        comparison_mark=mark_comp_processed,
        params=impression_params.comparison_params,
    )
    logger.debug("CMC is calculated")

    if not impression_params.skip_plots:
        save_impression_comparison_plots(
            mark_ref=mark_ref_processed,
            mark_comp=mark_comp_processed,
            cmc_result=cmc_result,
            comparison_params=impression_params.comparison_params,
            working_dir=vault.resource_path,
            files_to_save=ComparisonImpressionFiles,
            metadata_reference=impression_params.metadata_reference,
            metadata_compared=impression_params.metadata_compared,
        )
        logger.debug(f"images saved in:{vault.resource_path}")

    comparison_results = {
        "n_cells": cmc_result.cell_count,
        "score": cmc_result.cmc_count,
        "cmc_fraction": cmc_result.cmc_fraction,
        "cmc_area_fraction": cmc_result.cmc_area_fraction,
        "consensus_rotation": cmc_result.consensus_rotation,
        "consensus_translation": cmc_result.consensus_translation,
    }
    return ComparisonResponseImpression(
        urls=ComparisonResponseImpressionURL.from_enum(enum=ComparisonImpressionFiles, base_url=vault.access_url),
        cells=[cell.model_dump() for cell in cmc_result.cells],
        comparison_results=comparison_results,
    )


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_SCORE_STRIATION}",
    summary="Compare two striation profiles",
    description="""
    Reads preprocessed striation profiles from the comparison and reference directories,
    performs pairwise comparison, and calculates a score (correlation coefficient).
    The score, together with plots, are saved and made available via URLs.
    """,
    responses={
        HTTPStatus.NOT_FOUND: {"description": "mark or profile file not found"},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "invalid mark data"},
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
    if not striation_params.skip_plots:
        save_striation_comparison_plots(
            mark_ref=mark_ref,
            mark_comp=mark_comp,
            mark_correlations=comparison_result,
            working_dir=vault.resource_path,
            files_to_save=ComparisonStriationFiles,
            metadata_reference=striation_params.metadata_reference,
            metadata_compared=striation_params.metadata_compared,
        )
        logger.debug(f"images saved in:{vault.resource_path}")

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
        ComparisonStriationFiles.mark_reference_aligned_surface_map.get_file_path(vault.resource_path),
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(
        comparison_result.mark_reference_aligned.scan_image,
        ComparisonStriationFiles.mark_reference_aligned_preview.get_file_path(vault.resource_path),
    )
    surface_map_pipeline(
        comparison_result.mark_compared_aligned.scan_image,
        ComparisonStriationFiles.mark_compared_aligned_surface_map.get_file_path(vault.resource_path),
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(
        comparison_result.mark_compared_aligned.scan_image,
        ComparisonStriationFiles.mark_compared_aligned_preview.get_file_path(vault.resource_path),
    )
    logger.debug(f"marks saved in:{vault.resource_path}")

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
    responses={
        HTTPStatus.NOT_FOUND: {"description": "mark file or LR system not found"},
    },
)
async def calculate_lr_impression(lr_input: CalculateLRImpression) -> LRResponse:
    """Calculate likelihood ratio for impression mark comparison."""
    vault = create_vault(lr_input.tag)
    result = process_lr_impression(lr_input=lr_input, working_dir=vault.resource_path)
    return LRResponse(
        urls=LRResponseURL.from_enum(enum=LRFiles, base_url=vault.access_url),
        llr=result.log_lr,
        llr_lower_ci=result.log_lr_lower_ci,
        llr_upper_ci=result.log_lr_upper_ci,
    )


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_STRIATION}",
    summary="Calculate likelihood ratio for striation mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of striation marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
    responses={
        HTTPStatus.NOT_FOUND: {"description": "mark file or LR system not found"},
    },
)
async def calculate_lr_striation(lr_input: CalculateLRStriation) -> LRResponse:
    """Calculate likelihood ratio for striation mark comparison."""
    vault = create_vault(lr_input.tag)
    result = process_lr_striation(lr_input=lr_input, working_dir=vault.resource_path)
    return LRResponse(
        urls=LRResponseURL.from_enum(enum=LRFiles, base_url=vault.access_url),
        llr=result.log_lr,
        llr_lower_ci=result.log_lr_lower_ci,
        llr_upper_ci=result.log_lr_upper_ci,
    )
