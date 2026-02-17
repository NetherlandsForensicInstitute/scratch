from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import UUID4

from constants import RoutePrefix
from file_services import fetch_resource_path
from settings import SettingsDep

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
    Fetch a file from a vault directory using its unique token and filename.

    Supported file types: PNG images (.png) and X3P scan files (.x3p).
    Files must exist within the vault's storage directory and cannot be accessed outside of it.
    """,
    response_description="File content with media type image/png (.png) or application/octet-stream (.x3p).",
    responses={
        HTTPStatus.FORBIDDEN: {"description": "Access denied - path traversal detected"},
        HTTPStatus.NOT_FOUND: {"description": "File not found"},
    },
)
async def get_file(token: UUID4, filename: RelativePath, settings: SettingsDep) -> FileResponse:
    """
    Retrieve a processed file from vault storage.

    Fetches files (PNG images or X3P scans) from vault directories created by preprocessing
    endpoints. Validates that the requested file path is within the storage directory to
    prevent path traversal attacks.

    :param token: Unique vault directory token (UUID4).
    :param filename: Relative filename (must be .png or .x3p).
    :param settings: Application settings dependency.
    :return: FileResponse with appropriate media type (image/png or application/octet-stream).
    :raises HTTPException: 403 if path traversal detected, 404 if file not found.
    """
    filepath = fetch_resource_path(token) / filename
    if not filepath.resolve().is_relative_to(settings.storage.resolve()):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Access denied")

    if not filepath.exists():
        raise HTTPException(HTTPStatus.NOT_FOUND, f"File {filename} not found.")

    return FileResponse(
        path=filepath,
        media_type="image/png" if filename.suffix == ".png" else "application/octet-stream",
    )
