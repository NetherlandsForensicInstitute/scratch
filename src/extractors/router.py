from http import HTTPStatus
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import UUID4

from dependencies import get_tmp_dir

from .schemas import FileName

ROUTE = "/extractor"
extractor_route = APIRouter(prefix=ROUTE, tags=[ROUTE])


@extractor_route.get(
    path="/files/{token}/{filename}",
    summary="Return the file at the given filepath.",
    description="""
    given some file path returns the image located at the path.
    """,
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "Invalid token."},
        HTTPStatus.NOT_FOUND: {"description": "File not found"},
    },
)
async def get_file(
    token: UUID4,
    filename: FileName,
    temp_dir: Path = Depends(get_tmp_dir),
) -> FileResponse:
    """
    Get image from file path.

    This endpoint retrieves an image/scan file from a temporary directory and returns it as a FileResponse.

    :param token: Temporary directory token.
    :param filename: Name of the file to retrieve (must end with .png or .x3p).
    :param temp_dir: Temporary directory to store temporary files.
    :returns: FileResponse containing the requested image.
    """
    if not next(temp_dir.glob(str(token)), None):
        logger.error(f"Directory not found {temp_dir / str(token)}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Expired or invalid token: {token}")
    if not (filepath := temp_dir / str(token) / filename).exists():
        logger.error(f"File not found in temp dir: {filepath}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"File {filepath.name} not found.")

    return FileResponse(
        path=filepath,
        media_type="image/png" if filename.endswith(".png") else "application/octet-stream",
    )
