from conversion.export.mark import load_mark_from_path
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

from constants import LIGHT_SOURCES, OBSERVER, ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation, LRResponse, LRResponseURL
from file_services import create_vault
from models import DirectoryAccess
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from processors.controller import calculate_striation_plots, save_plots
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
    expected_files = ComparisonResponseStriation.get_files(vault.resource_path)
    mark_ref = load_mark_from_path(path=striation_params.mark_ref, stem="processed")
    mark_comp = load_mark_from_path(path=striation_params.mark_comp, stem="processed")
    logger.debug("Marking loaded")
    mark_comparison = calculate_striation_plots(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        ref_path=striation_params.mark_ref,
        compare_path=striation_params.mark_ref,
    )
    save_plots(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_correlations=mark_comparison,
        files_to_save=expected_files,
        meta_data_ref=striation_params.param.metadata_reference,
        meta_data_compare=striation_params.param.metadata_compared,
    )
    surface_map_pipeline(
        mark_comparison.mark_reference_aligned.scan_image,
        expected_files["mark_ref_surfacemap"],
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(mark_comparison.mark_reference_aligned.scan_image, expected_files["mark_ref_depthmap"])
    surface_map_pipeline(
        mark_comparison.mark_compared_aligned.scan_image,
        expected_files["mark_comp_surfacemap"],
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(mark_comparison.mark_compared_aligned.scan_image, expected_files["mark_comp_depthmap"])
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
