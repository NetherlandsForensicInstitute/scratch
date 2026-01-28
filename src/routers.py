from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from comparators.router import comparison_router
from extractors.router import extractor_route
from preprocessors import preprocessor_route
from processors.router import processors

prefix_router = APIRouter()

prefix_router.include_router(preprocessor_route)
prefix_router.include_router(processors)
prefix_router.include_router(comparison_router)
prefix_router.include_router(extractor_route)


@prefix_router.get(
    path="/",
    summary="Redirect to API documentation",
    description="Redirects to the interactive API documentation.",
    include_in_schema=False,
)
async def root() -> RedirectResponse:
    """
    Redirect to the API documentation.

    This endpoint redirects users to the interactive API documentation at /docs.

    :return: RedirectResponse to the API documentation.
    """
    return RedirectResponse(url="/docs")
