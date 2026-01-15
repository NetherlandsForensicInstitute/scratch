from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from constants import RoutePrefix

comparison_router = APIRouter(
    prefix=f"/{RoutePrefix.COMPARATOR}",
    tags=[RoutePrefix.COMPARATOR],
)


@comparison_router.get(
    path="",
    summary="Redirect to comparator documentation",
    description="""Redirects to the comparator section in the API documentation.""",
    include_in_schema=False,
)
async def comparator_root() -> RedirectResponse:
    """
    Redirect to the comparator section in Swagger docs.

    This endpoint redirects users to the comparator tag section in the
    interactive API documentation at /docs.

    :return: RedirectResponse to the comparator documentation section.
    """
    return RedirectResponse(url=f"/docs#operations-tag-{RoutePrefix.COMPARATOR}")
