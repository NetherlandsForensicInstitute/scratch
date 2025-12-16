from http import HTTPStatus

from fastapi import APIRouter

from .pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline, x3p_pipeline
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
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def process_scan(upload_scan: UploadScan) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the output directory.
    """
    parsed_scan = parse_scan_pipeline(upload_scan.scan_file, upload_scan.parameters)
    return ProcessedDataLocation(
        x3p_image=x3p_pipeline(parsed_scan, upload_scan.x3p_path),
        surfacemap_image=surface_map_pipeline(parsed_scan, upload_scan.surfacemap_path, upload_scan.parameters),
        preview_image=preview_pipeline(parsed_scan, upload_scan.preview_path),
    )
