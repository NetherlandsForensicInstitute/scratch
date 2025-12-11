from http import HTTPStatus
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import UUID4, HttpUrl

from constants import BASE_URL
from dependencies import get_tmp_dir, get_token

from .pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline, x3p_pipeline
from .schemas import FileName, ProcessedDataLocation, UploadScan

ROUTE = "/preprocessor"
preprocessor_route = APIRouter(prefix=ROUTE, tags=[ROUTE])


@preprocessor_route.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of pre-processors endpoint, you can use basic **markup**""",
)
async def comparison_root() -> dict[str, str]:
    """Fetch a simple message from the REST API.

    Here is some more information about the function some notes what is expected.
    Special remarks what the function is doing.

    return: dict[str,str] but, use as much as possible Pydantic for return types
    """
    return {"message": "Hello from the pre-processors"}


@preprocessor_route.post(
    path="/process-scan",
    summary="Create surface_map and preview image from the scan file.",
    description="""
    Processes the scan file from the given filepath and generates several derived outputs, including
    an X3P file, a preview image, and a surface map, these files are saved to the output directory given as parameter.
    The endpoint parses and validates the file before running the processing pipeline.
""",
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def process_scan(
    upload_scan: UploadScan,
    temp_dir: Path = Depends(get_tmp_dir),
    token: UUID = Depends(get_token),
) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to an temp directiory and returns urls to retrieve them.
    """
    token_str = str(token)
    output_dir = temp_dir / token_str
    output_dir.mkdir(parents=True, exist_ok=True)
    base_image_url = f"{BASE_URL}{ROUTE}/file/{token_str}"

    parsed_scan = parse_scan_pipeline(upload_scan.scan_file, upload_scan.parameters)
    x3p_pipeline(parsed_scan, output_dir / upload_scan.x3p_filename)
    surface_map_pipeline(parsed_scan, output_dir / upload_scan.surfacemap_filename, upload_scan.parameters)
    preview_pipeline(parsed_scan, output_dir / upload_scan.preview_filename)

    logger.info(f"Generated files saved to {temp_dir}")
    return ProcessedDataLocation(
        x3p_image=HttpUrl(f"{base_image_url}/{upload_scan.x3p_filename}"),
        preview_image=HttpUrl(f"{base_image_url}/{upload_scan.preview_filename}"),
        surfacemap_image=HttpUrl(f"{base_image_url}/{upload_scan.surfacemap_filename}"),
    )


@preprocessor_route.get(
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
