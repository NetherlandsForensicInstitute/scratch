import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from image_generation.image_generation import compute_3d_image, get_array_for_display
from loguru import logger
from parsers import load_scan_image
from parsers.exceptions import ExportError
from parsers.x3p import save_to_x3p
from pydantic import HttpUrl

from constants import BASE_URL
from dependencies import get_tmp_dir
from preprocessors.helpers import export_image_pipeline
from preprocessors.models import ErrorImageGenerationModel, ParsingError

from .schemas import ProcessedDataLocation, UploadScan

preprocessor_route = APIRouter(
    prefix="/preprocessor",
    tags=["preprocessor"],
)


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
        400: {"description": "parse error", "model": ParsingError},
        500: {
            "description": "image generation error",
            "model": ErrorImageGenerationModel,
        },
    },
)
async def process_scan(upload_scan: UploadScan, temp_dir: Path = Depends(get_tmp_dir)) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the output directory.
    """
    token = temp_dir.name
    base_image_url = f"{BASE_URL}/preprocessor/image_file/{token}"
    logger.debug(f"Processing scan file with url:{base_image_url}")
    scan_file_path = temp_dir / "scan.x3p"
    parsed_scan = load_scan_image(upload_scan.scan_file).subsample(step_x=1, step_y=1)
    try:
        save_to_x3p(image=parsed_scan, output_path=scan_file_path)
    except ExportError as err:
        logger.error(f"Exporting x3p failed to path:{scan_file_path}, from error:{str(err)}")
        raise HTTPException(status_code=500, detail=f"Failed to save the scan file  {scan_file_path}: {str(err)}")
    export_image_pipeline(
        file_path=temp_dir / "surface_map.png", image_generator=compute_3d_image, scan_image=parsed_scan
    )
    export_image_pipeline(
        file_path=temp_dir / "preview.png", image_generator=get_array_for_display, scan_image=parsed_scan
    )
    logger.info(f"Generated files saved to {temp_dir}")
    return ProcessedDataLocation(
        x3p_image=HttpUrl(f"{base_image_url}/scan.x3p"),
        preview_image=HttpUrl(f"{base_image_url}/preview.png"),
        surfacemap_image=HttpUrl(f"{base_image_url}/surface_map.png"),
    )


@preprocessor_route.get(
    path="/image_file/{token}/{file_name}",
    summary="Giving an path returns a image.",
    description="""
    given some file path returns the image located at the path.
    """,
)
async def get_image(token: str, file_name: str) -> FileResponse:
    """Get image from file path."""
    temp_dir = Path(tempfile.gettempdir()) / token
    if not temp_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Temp dir {temp_dir} not found.")
    if not (temp_dir / file_name).exists():
        raise HTTPException(status_code=404, detail=f"File {file_name} not found in temp dir.")
    if not file_name.endswith((".png", ".x3p")):
        raise HTTPException(status_code=400, detail="Unsupported file type requested.")
    return FileResponse(
        path=Path(f"{temp_dir}/{file_name}"),
        media_type="image/png" if file_name.endswith(".png") else "application/octet-stream",
    )
