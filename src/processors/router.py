from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from constants import ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation, LRResponse
from models import DirectoryAccess
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
    impression.mark_comp
    impression.mark_ref
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
async def calculate_score_striation(striation: CalculateScoreStriation) -> ComparisonResponseStriation:
    """Compare two striation profiles."""
    striation.mark_comp
    striation.mark_ref
    vault = DirectoryAccess()  # type: ignore
    return ComparisonResponseStriation.generate_urls(vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_IMPRESSION}",
    summary="",
    description="""""",
)
async def calculate_lr_impression(striation: CalculateLRImpression) -> LRResponse:
    """TODO."""
    striation.mark_comp
    striation.mark_ref
    vault = DirectoryAccess()  # type: ignore
    # TODO::
    # - when, where, and how is the LRParams used?
    # - create controllers module
    # - This section below need to be moved to controllers.py
    # - Does calculate_lr return a lr float value or do I need to get that from lr_system?
    # - where am I importing the lr functions from? library? we need to write it?
    #
    # controllers.py
    # def compute_n_plot_lr(ref: Mark, comp: Mark, score: int: lr_system: Path) -> float:
    #     system=get_lr_system(lr_system)
    #     lr = calculate_lr(  # NOTE: I am assuming this returns an LR value
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
    return LRResponse.generate_urls(vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_LR_STRIATION}",
    summary="",
    description="""""",
)
async def calculate_lr_striation(striation: CalculateLRStriation) -> LRResponse:
    """TODO."""
    striation.mark_comp
    striation.mark_ref
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
    return LRResponse.generate_urls(vault.access_url)
