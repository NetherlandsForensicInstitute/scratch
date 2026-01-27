from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from constants import RoutePrefix

processors = APIRouter(
    prefix=f"/{RoutePrefix.PROCESSOR}",
    tags=[RoutePrefix.PROCESSOR],
)


@processors.get(
    path="",
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
