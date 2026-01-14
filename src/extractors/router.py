from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import UUID4

from constants import EXTRACTOR_ROUTE
from file_services import fetch_resource_path
from settings import SettingsDep

from .schemas import RelativePath

extractor_route = APIRouter(prefix=EXTRACTOR_ROUTE, tags=[EXTRACTOR_ROUTE])


@extractor_route.get(
    path="/",
    summary="Health check for extractor service",
    description="""Returns a simple status message to verify the extractor service is running.""",
)
async def extractor_root() -> dict[str, str]:
    """
    Health check endpoint for the extractor service.

    Returns a simple greeting message to confirm the extractor API is accessible.

    :return: Dictionary containing a status message.
    """
    return {"message": "Hello from the extractors"}


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
async def get_file(token: UUID4, filename: RelativePath, settings: SettingsDep) -> FileResponse:
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
    filepath = fetch_resource_path(token) / filename
    if not filepath.resolve().is_relative_to(settings.storage.resolve()):
        raise HTTPException(HTTPStatus.FORBIDDEN, "Access denied")

    if not filepath.exists():
        raise HTTPException(HTTPStatus.NOT_FOUND, f"File {filename} not found.")

    return FileResponse(
        path=filepath,
        media_type="image/png" if filename.suffix == ".png" else "application/octet-stream",
    )
