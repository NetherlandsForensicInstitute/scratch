from http import HTTPStatus

from fastapi import APIRouter
from loguru import logger
from pydantic import UUID4

from constants import PREPROCESSOR_ROUTE
from file_services import create_vault, fetch_directory_access, fetch_resource_file, generate_files, generate_urls
from models import ProcessDataUrls

from .pipelines import parse_scan_pipeline, preview_pipeline, surface_map_pipeline, x3p_pipeline
from .schemas import EditImage, EditImageParameters, ProcessScanOutput, UploadScan

preprocessor_route = APIRouter(prefix=PREPROCESSOR_ROUTE, tags=[PREPROCESSOR_ROUTE])


@preprocessor_route.get(
    path="/",
    summary="check status of comparison proces",
    description="""Some description of pre-processors endpoint, you can use basic **markup**""",
)
async def preprocessor_root() -> dict[str, str]:
    """
    Fetch a simple message from the REST API.

    Here is some more information about the function some notes what is expected.
    Special remarks what the function is doing.

    :return: Use as much as possible Pydantic for return types.
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
async def process_scan(upload_scan: UploadScan) -> ProcessScanOutput:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to a temp directory and returns urls to retrieve them.

    :param upload_scan: The uploaded scan data and parameters.
    :return: Access URLs for the generated files.
    """
    parsed_scan = parse_scan_pipeline(upload_scan.scan_file, upload_scan.parameters)
    vault = create_vault(upload_scan.tag)
    files = generate_files(vault.resource_path, scan="scan.x3p", preview="preview.png", surface_map="surface_map.png")
    scan = x3p_pipeline(parsed_scan, files["scan"])
    surface_map = surface_map_pipeline(parsed_scan, files["surface_map"], upload_scan.parameters)
    preview = preview_pipeline(parsed_scan, files["preview"])

    logger.info(f"Generated files saved to {vault}")
    return ProcessScanOutput(
        downloads=generate_urls(vault.access_url, scan, preview, surface_map),  # type: ignore
        token=vault.token,
    )


@preprocessor_route.post(
    path="/edit-scan",
    summary="Validate and parse a scan file with edit parameters.",
    description="""
    Parse and validate a scan file (X3P format only) with the provided edit parameters
    (mask, crop, subsampling). Creates a new vault for storing future outputs.

    Note: Image generation is currently not implemented.
""",
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "parse error"},
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "processing error",
        },
    },
)
async def edit_scan(edit_image: EditImage) -> ProcessScanOutput:
    """
    Validate and parse a scan file with edit parameters.

    Accepts an X3P scan file and edit parameters (mask, zoom, step sizes),
    validates the file format, parses it according to the parameters, and
    creates a vault directory for future outputs. Returns access URLs for the vault.
    """
    _ = parse_scan_pipeline(edit_image.scan_file, edit_image.parameters)
    vault = create_vault(edit_image.tag)

    logger.info(f"Generated files saved to {vault}")
    return ProcessScanOutput(
        downloads=generate_urls(vault.access_url),  # type: ignore
        token=vault.token,
    )


@preprocessor_route.post(
    path="/edit-scans/{token}",
    summary="Re-process an existing scan file with new edit parameters.",
    description="""
    Retrieve a previously uploaded scan file using its token, tag, and filename,
    then parse it with new edit parameters (mask, zoom, subsampling).

    If overwrite=False, creates a new vault for storing outputs. If overwrite=True,
    reuses the existing vault directory.

    Note: Image generation is currently not implemented.
""",
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "parse error"},
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "processing error",
        },
    },
)
async def edit_existing_scan(token: UUID4, parameters: EditImageParameters) -> ProcessDataUrls:
    """
    Re-process an existing scan file with new edit parameters.

    Fetches a previously uploaded scan file from the vault identified by token/tag/filename,
    parses it with the new edit parameters (mask, zoom, step sizes), and creates a new
    vault directory (unless overwrite=True). Returns access URLs for the vault.
    """
    vault = fetch_directory_access(token)
    _ = parse_scan_pipeline(fetch_resource_file(vault.resource_path, "scan.x3p"), parameters)
    if not parameters.overwrite:
        vault = create_vault(vault.tag)

    logger.info(f"Generated files saved to {vault}")
    return ProcessDataUrls.model_validate(generate_urls(vault.access_url))
