import numpy as np
from computations.likelihood_ratio import calculate_lr_impression as _calculate_lr_impression
from computations.likelihood_ratio import calculate_lr_striation as _calculate_lr_striation
from computations.likelihood_ratio import get_lr_system, get_reference_data
from conversion.export.mark import load_mark_from_path
from conversion.export.profile import load_profile_from_path
from conversion.plots.data_formats import ImpressionComparisonMetrics
from conversion.plots.utils import build_results_metadata_impression, build_results_metadata_striation
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

from constants import LIGHT_SOURCES, OBSERVER, ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation, LRResponse, LRResponseURL
from file_services import create_vault
from preprocessors.pipelines import preview_pipeline, surface_map_pipeline
from processors.controller import (
    compare_striation_marks,
    save_lr_impression_plot,
    save_lr_striation_plot,
    save_striation_comparison_plots,
)
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
    vault = create_vault(impression.tag)
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
        files_to_save=expected_files,
        metadata_reference=striation_params.param.metadata_reference,
        metadata_compared=striation_params.param.metadata_compared,
    )
    surface_map_pipeline(
        comparison_result.mark_reference_aligned.scan_image,
        expected_files["mark_ref_surfacemap"],
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(comparison_result.mark_reference_aligned.scan_image, expected_files["mark_ref_preview"])
    surface_map_pipeline(
        comparison_result.mark_compared_aligned.scan_image,
        expected_files["mark_comp_surfacemap"],
        LIGHT_SOURCES,
        OBSERVER,
    )
    preview_pipeline(comparison_result.mark_compared_aligned.scan_image, expected_files["mark_comp_preview"])
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
async def calculate_lr_impression(lr_input: CalculateLRImpression) -> LRResponse:
    """Calculate likelihood ratio for impression mark comparison."""
    vault = create_vault(lr_input.tag)
    files = LRResponseURL.get_files(vault.resource_path)

    lr_system = get_lr_system(lr_input.lr_system_path)
    reference_data = get_reference_data(lr_input.lr_system_path)

    llr_data = _calculate_lr_impression(lr_system, lr_input.score, lr_input.n_cells)
    metrics = ImpressionComparisonMetrics(
        area_correlation=0.82,
        cell_correlations=np.array([
            [0.75, 0.88, 0.45],
            [0.92, 0.31, 0.67],
        ]),
        cmc_score=66.7,
        mean_square_ref=1.25,
        mean_square_comp=1.31,
        mean_square_of_difference=0.42,
        has_area_results=True,
        has_cell_results=True,
        cell_positions_compared=np.array([
            [10.0, 20.0],
            [10.0, 60.0],
            [10.0, 100.0],
            [50.0, 20.0],
            [50.0, 60.0],
            [50.0, 100.0],
        ]),
        cell_rotations_compared=np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02]),
        cmc_area_fraction=55.0,
        cutoff_low_pass=250.0,
        cutoff_high_pass=25.0,
        cell_size_um=300.0,
        max_error_cell_position=50.0,
        max_error_cell_angle=3.0,
        cell_similarity_threshold=0.25,
    )  # TODO replace with saved list[Cells] instead when implemented
    results_metadata = {
        "Date report": "2023-02-16",
        "User ID": "RUHES (apc_abal)",
        "Mark type": "Aperture shear striation",
        "Score type": "CCF",
        "Score (transform)": "0.97 (1.86)",
        "LogLR (5%, 95%)": "5.19 (5.17, 5.24)",
        "# of KM scores": "1144",
        "# of KNM scores": "296462",
    }
    mark_ref = load_mark_from_path(lr_input.mark_dir_ref, stem="processed")
    mark_comp = load_mark_from_path(lr_input.mark_dir_comp, stem="processed")
    build_results_metadata_impression(
        reference_data=reference_data,
        llr_data=llr_data,
        date_report=lr_input.date_report,
        user_id=lr_input.user_id,
        mark_type=mark_ref.mark_type.value,
        score=lr_input.score,
        n_cells=lr_input.n_cells,
        knm_model=reference_data.knm_model,
        km_model=reference_data.km_model,
    )
    save_lr_impression_plot(
        reference_data=reference_data,
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        metrics=metrics,
        metadata_reference=lr_input.metadata_reference,
        metadata_compared=lr_input.metadata_compared,
        results_metadata=results_metadata,
        score=lr_input.score,
        lr=llr_data.llrs[0],
        output_path=files["lr_overview_plot"],
    )
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=llr_data.llrs[0])


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_STRIATION}",
    summary="Calculate likelihood ratio for striation mark comparison",
    description="""
    Calculates a likelihood ratio (LR) for a pair of striation marks
    using the provided score and path to the LR system.
    The LR value, together with plots, are saved and made available via URLs.
    """,
)
async def calculate_lr_striation(lr_input: CalculateLRStriation) -> LRResponse:
    """Calculate likelihood ratio for striation mark comparison."""
    vault = create_vault(lr_input.tag)
    files = LRResponseURL.get_files(vault.resource_path)

    lr_system = get_lr_system(lr_input.lr_system_path)
    reference_data = get_reference_data(lr_input.lr_system_path)

    llr_data = _calculate_lr_striation(lr_system, lr_input.score)
    mark_ref = load_mark_from_path(lr_input.mark_dir_ref, stem="processed")
    mark_ref_aligned = load_mark_from_path(lr_input.mark_dir_ref, stem="processed")  # todo make aligned
    mark_comp = load_mark_from_path(lr_input.mark_dir_comp, stem="processed")
    mark_comp_aligned = load_mark_from_path(lr_input.mark_dir_comp, stem="processed")  # todo make aligned
    results_metadata = build_results_metadata_striation(
        reference_data=reference_data,
        llr_data=llr_data,
        date_report=lr_input.date_report,
        user_id=lr_input.user_id,
        mark_type=mark_ref.mark_type.value,
        score=lr_input.score,
        score_transform=lr_input.score,
    )

    save_lr_striation_plot(
        mark_ref=mark_ref,
        mark_comp=mark_comp,
        mark_ref_aligned=mark_ref_aligned,
        mark_comp_aligned=mark_comp_aligned,
        output_path=files["lr_overview_plot"],
        metadata_reference=lr_input.metadata_reference,
        metadata_compared=lr_input.metadata_compared,
        results_metadata=results_metadata,
        score=lr_input.score,
        lr=llr_data.llrs[0],
        reference_data=reference_data,
    )
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=llr_data.llrs[0])
