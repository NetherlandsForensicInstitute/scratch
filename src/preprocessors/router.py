from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from loguru import logger

from constants import PreprocessorEndpoint, RoutePrefix
from extractors import ProcessedDataAccess
from extractors.schemas import PrepareMarkResponseImpression, PrepareMarkResponseStriation
from file_services import create_vault
from preprocessors.controller import process_prepare_mark, process_scan_controller

from .schemas import EditImage, PrepareMarkImpression, PrepareMarkStriation, UploadScan

preprocessor_route = APIRouter(prefix=f"/{RoutePrefix.PREPROCESSOR}", tags=[RoutePrefix.PREPROCESSOR])


@preprocessor_route.get(
    path=PreprocessorEndpoint.ROOT,
    summary="Redirect to preprocessor documentation",
    description="""Redirects to the preprocessor section in the API documentation.""",
    include_in_schema=False,
)
async def preprocessor_root() -> RedirectResponse:
    """
    Redirect to the preprocessor section in Swagger docs.

    This endpoint redirects users to the preprocessor tag section in the
    interactive API documentation at /docs.

    :return: RedirectResponse to the preprocessor documentation section.
    """
    return RedirectResponse(url=f"/docs#operations-tag-{RoutePrefix.PREPROCESSOR}")


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.PROCESS_SCAN}",
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
async def process_scan(upload_scan: UploadScan) -> ProcessedDataAccess:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to a temp directory and returns urls to retrieve them.

    :param upload_scan: The uploaded scan data and parameters.
    :return: Access URLs for the generated files.
    """
    vault = create_vault(upload_scan.tag)
    process_scan_controller(
        scan_file=upload_scan.scan_file,
        output_path=vault.resource_path,
        light_sources=upload_scan.light_vectors,
        observer=upload_scan.observer.unit_vector,
    )
    logger.info(f"Generated files saved to {vault}")
    return ProcessedDataAccess.generate_urls(vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.PREPARE_MARK_IMPRESSION}",
    summary="Preprocess a scan into analysis-ready mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for impression marks.

    Outputs two processed mark representations (.npz data and .json
    metadata) saved to the vault, returning URLs for file access.
    """,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def prepare_mark_impression(prepare_mark_parameters: PrepareMarkImpression) -> PrepareMarkResponseImpression:
    """Prepare the ScanFile, save it to the vault and return the urls to acces the files."""
    vault = create_vault(prepare_mark_parameters.tag)
    process_prepare_mark(
        scan_file=prepare_mark_parameters.scan_file,
        files=PrepareMarkResponseImpression.get_files(vault.resource_path),
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseImpression.generate_urls(vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.PREPARE_MARK_STRIATION}",
    summary="Preprocess a scan into analysis-ready mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for striation marks.

    Outputs two processed mark representations (.npz data and .json
    metadata) saved to the vault, returning URLs for file access.
    """,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def prepare_mark_striation(prepare_mark_parameters: PrepareMarkStriation) -> PrepareMarkResponseStriation:
    """Prepare the ScanFile, save it to the vault and return the urls to acces the files."""
    vault = create_vault(prepare_mark_parameters.tag)
    process_prepare_mark(
        files=PrepareMarkResponseStriation.get_files(vault.resource_path),
        scan_file=prepare_mark_parameters.scan_file,
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseStriation.generate_urls(vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.EDIT_SCAN}",
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
async def edit_scan(edit_image: EditImage) -> ProcessedDataAccess:
    """
    Validate and parse a scan file with edit parameters.

    Accepts an X3P scan file and edit parameters (mask, zoom, step sizes),
    validates the file format, parses it according to the parameters, and
    creates a vault directory for future outputs. Returns access URLs for the vault.
    """
    vault = create_vault(edit_image.tag)

    logger.info(f"Generated files saved to {vault}")
    return ProcessedDataAccess.generate_urls(vault.access_url)
