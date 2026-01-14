from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import UUID4

from constants import RoutePrefix
from file_services import fetch_directory_access, fetch_resource_file

from .schemas import RelativePath

extractor_route = APIRouter(prefix=f"/{RoutePrefix.EXTRACTOR}", tags=[RoutePrefix.EXTRACTOR])


@extractor_route.get(
    path="",
    summary="Redirect to extractor documentation",
    description="""Redirects to the extractor section in the API documentation.""",
    include_in_schema=False,
)
async def extractor_root() -> RedirectResponse:
    """
    Redirect to the extractor section in Swagger docs.

    This endpoint redirects users to the extractor tag section in the
    interactive API documentation at /docs.

    :return: RedirectResponse to the extractor documentation section.
    """
    return RedirectResponse(url=f"/docs#operations-tag-{RoutePrefix.EXTRACTOR}")


@extractor_route.get(
    path="/files/{token}/{filename}",
    summary="Retrieve a processed file from vault storage.",
    description="""
    Fetch a file from a vault directory using its unique token, project tag, and filename.

    Supported file types: PNG images (.png) and X3P scan files (.x3p).
    Files must exist within the vault's storage directory and cannot be accessed outside of it.
    """,
    responses={
        HTTPStatus.FORBIDDEN: {"description": "Access denied - path traversal detected"},
        HTTPStatus.NOT_FOUND: {"description": "File not found"},
    },
)
async def get_file(token: UUID4, filename: RelativePath) -> FileResponse:
    """
    Retrieve a processed file from vault storage.

    Fetches files (PNG images or X3P scans) from vault directories created by preprocessing
    endpoints. Validates that the requested file path is within the storage directory to
    prevent path traversal attacks.

    :param token: Unique vault directory token (UUID4).
    :param tag: Project tag identifier.
    :param filename: Relative filename (must be .png or .x3p).
    :param settings: Application settings dependency.
    :return: FileResponse with appropriate media type (image/png or application/octet-stream).
    :raises HTTPException: 403 if path traversal detected, 404 if file not found.
    """
    return FileResponse(
        path=fetch_resource_file(fetch_directory_access(token).resource_path, filename),
        media_type="image/png" if filename.suffix == ".png" else "application/octet-stream",
    )
