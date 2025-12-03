from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from image_generation.image_generation import generate_3d_image, get_array_for_display
from loguru import logger
from parsers import from_file
from parsers.exceptions import ExportError
from parsers.x3p import save_to_x3p

from preprocessors.models import ImageGenerationError, ParsingError
from src.preprocessors.helpers import export_image_pipeline

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
            "model": ImageGenerationError,
        },
    },
)
async def process_scan(upload_scan: UploadScan) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the output directory.
    """
    surface_image_path = upload_scan.output_dir / "surface_map.png"
    preview_image_path = upload_scan.output_dir / "preview.png"
    scan_file_path = upload_scan.output_dir / "scan.x3p"
    parsed_scan = from_file(upload_scan.scan_file).subsample_data(step_x=1, step_y=1)
    try:
        save_to_x3p(image=parsed_scan, output_path=scan_file_path)
    except ExportError as err:
        logger.error("jammer man, failed to save the scan file")
        raise HTTPException(status_code=500, detail=f"Failed to save the scan file  {scan_file_path}: {str(err)}")
    export_image_pipeline(file_path=surface_image_path, image_generator=generate_3d_image, scan_image=parsed_scan)
    export_image_pipeline(file_path=preview_image_path, image_generator=get_array_for_display, scan_image=parsed_scan)
    return ProcessedDataLocation(
        x3p_image=scan_file_path,
        preview_image=preview_image_path,
        surfacemap_image=surface_image_path,
    )
