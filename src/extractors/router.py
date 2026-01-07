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
    path="/file/{token}/{file_name}",
    summary="Giving an path returns a image.",
    description="""
    given some file path returns the image located at the path.
    """,
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "Unsupported file type requested."},
        HTTPStatus.BAD_REQUEST: {"description": "Invalid token."},
        HTTPStatus.NOT_FOUND: {"description": "File not found"},
    },
)
async def get_file(
    token: UUID4,
    file_name: FileName,
    temp_dir: Path = Depends(get_tmp_dir),
) -> FileResponse:
    """
    Get image from file path.

    This endpoint retrieves an image/scan file from a temporary directory and returns it as a FileResponse.

    :param token: Temporary directory token.
    :param file_name: Name of the file to retrieve (must end with .png or .x3p).
    :param temp_dir: Temporary directory to store temporary files.
    :returns: FileResponse containing the requested image.
    """
    if not next(temp_dir.glob(str(token)), None):
        logger.error(f"Directory not found {temp_dir}/{token}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Expired or invalid token: {token}")
    if not (file_path := temp_dir / str(token) / file_name).exists():
        logger.error(f"File not found in temp dir: {file_path}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"File {file_path.name} not found.")

    return FileResponse(
        path=file_path,
        media_type="image/png" if file_name.endswith(".png") else "application/octet-stream",
    )
