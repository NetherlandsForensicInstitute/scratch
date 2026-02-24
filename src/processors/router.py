from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from constants import ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation, LRResponse, LRResponseURL
from models import DirectoryAccess
from processors.controller import calculate_lr, get_lr_system
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
async def calculate_score_striation(striation: CalculateScoreStriation) -> ComparisonResponseStriation:
    """Compare two striation profiles."""
    vault = DirectoryAccess()  # type: ignore
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
    lr_system = get_lr_system(impression.lr_system)
    lr = calculate_lr(impression.score, lr_system)
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=lr)


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
    """Calculate likelihood ratio for striation mark comparison."""
    vault = DirectoryAccess()  # type: ignore
    lr_system = get_lr_system(striation.lr_system)
    lr = calculate_lr(striation.score, lr_system)
    return LRResponse(urls=LRResponseURL.generate_urls(vault.access_url), lr=lr)
