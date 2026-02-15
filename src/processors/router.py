from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from constants import ProcessorEndpoint, RoutePrefix
from extractors.schemas import ComparisonResponseImpression, ComparisonResponseStriation
from models import DirectoryAccess
from processors.schemas import CalculateScoreImpression, CalculateScoreStriation

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
    summary="",
    description="""""",
    include_in_schema=False,
)
async def calculate_score_impression(impression: CalculateScoreImpression) -> ComparisonResponseImpression:
    """TODO."""
    # TODO: Would be better if we have the token
    # then the vault can be recreated
    # and comp and ref can be extracted from vault
    impression.mark_dir_comp
    impression.mark_dir_ref
    vault = DirectoryAccess()  # type: ignore
    return ComparisonResponseImpression.generate_urls(vault.access_url)


@processors.post(
    path=f"/{ProcessorEndpoint.CALCULATE_SCORE_STRIATION}",
    summary="",
    description="""""",
)
async def calculate_score_striation(striation: CalculateScoreStriation) -> ComparisonResponseStriation:
    """TODO."""
    # TODO: Would be better if we have the token
    # then the vault can be recreated
    # and comp and ref can be extracted from vault
    striation.mark_dir_comp
    striation.mark_dir_ref
    vault = DirectoryAccess()  # type: ignore
    return ComparisonResponseStriation.generate_urls(vault.access_url)
