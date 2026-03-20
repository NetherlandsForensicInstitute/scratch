from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from loguru import logger
from pydantic import BaseModel, Json

from constants import (
    LIGHT_SOURCES,
    OBSERVER,
    PreprocessorEndpoint,
    RoutePrefix,
)
from response_constants import (
    GeneratedImageFiles,
    PrepareMarkImpressionFiles,
    PrepareMarkStriationFiles,
    ProcessFiles,
)
from file_services import create_vault
from preprocessors.controller import edit_scan_image, process_prepare_impression_mark, process_prepare_striation_mark
from response_models import (
    GeneratedImages,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    ProcessedDataAccess,
)

from .exceptions import ArrayShapeMismatchError
from .pipelines import (
    parse_mask_pipeline,
    parse_scan_pipeline,
    preview_pipeline,
    surface_map_pipeline,
    x3p_pipeline,
)
from .schemas import EditImage, PrepareMarkImpression, PrepareMarkStriation, UploadScan

preprocessor_route = APIRouter(prefix=f"/{RoutePrefix.PREPROCESSOR}", tags=[RoutePrefix.PREPROCESSOR])


def _generate_openapi_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate example fields in the Swagger docs for endpoints receiving multipart/form-data with a binary mask."""
    return {
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "properties": {
                            "params": model.model_json_schema(),
                            "mask_data": {"type": "string", "format": "binary", "example": b"\x01\x00\x00\x01"},
                        },
                        "required": ["params", "mask_data"],
                    }
                },
                "application/json": {
                    "schema": {
                        "properties": {
                            "params": model.model_json_schema(),
                        },
                        "required": ["params"],
                    }
                },
            }
        }
    }


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
    Processes the scan file from the given filepath and generates several derived outputs:
    an X3P file, a preview image, and a surface map. The files are saved to an
    auto-generated vault and the response contains download URLs for each output.
    The endpoint parses and validates the file before running the processing pipeline.
""",
    response_description="Download URLs for the generated X3P scan, preview image, and surface map.",
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
    parsed_scan = parse_scan_pipeline(upload_scan.scan_file, upload_scan.step_size, upload_scan.step_size)
    x3p_pipeline(parsed_scan, ProcessFiles.scan_image.get_file_path(vault.resource_path))
    surface_map_pipeline(
        parsed_scan, ProcessFiles.surface_map_image.get_file_path(vault.resource_path), LIGHT_SOURCES, OBSERVER
    )
    preview_pipeline(parsed_scan, ProcessFiles.preview_image.get_file_path(vault.resource_path))

    logger.info(f"Generated files saved to {vault}")
    return ProcessedDataAccess.from_enum(enum=ProcessFiles, base_url=vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.PREPARE_MARK_IMPRESSION}",
    summary="Preprocess a scan into analysis-ready impression mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for impression marks.

    Outputs two processed mark representations (.npz data and .json
    metadata) saved to the vault, returning URLs for file access.

    The mask must have exactly the same shape (height × width) as the parsed scan image.
    """,
    responses={
        HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "mask shape does not match image shape"},
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
    openapi_extra=_generate_openapi_schema(model=PrepareMarkImpression),
)
async def prepare_mark_impression(
    params: Annotated[Json[PrepareMarkImpression], Form(...)], mask_data: bytes = File(...)
) -> PrepareMarkResponseImpression:
    """Prepare the ScanFile, save it to the vault and return the URLs to access the files."""
    vault = create_vault(params.tag)
    parsed_image = parse_scan_pipeline(params.scan_file, 1, 1)

    try:
        parsed_mask = parse_mask_pipeline(
            raw_data=mask_data,
            shape=parsed_image.data.shape,
            is_bitpacked=params.mask_is_bitpacked,
        )
    except ArrayShapeMismatchError as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e))

    process_prepare_impression_mark(
        scan_image=parsed_image,
        mark_type=params.mark_type,
        mask=parsed_mask,
        bounding_box=params.bounding_box,
        preprocess_parameters=params.mark_parameters,
        working_dir=vault.resource_path,
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseImpression.from_enum(enum=PrepareMarkImpressionFiles, base_url=vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.PREPARE_MARK_STRIATION}",
    summary="Preprocess a scan into analysis-ready striation mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for striation marks.

    Outputs two processed mark representations (.npz data and .json
    metadata) saved to the vault, returning URLs for file access.

    The mask must have exactly the same shape (height × width) as the parsed scan image.
    """,
    responses={
        HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "mask shape does not match image shape"},
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
    openapi_extra=_generate_openapi_schema(model=PrepareMarkStriation),
)
async def prepare_mark_striation(
    params: Annotated[Json[PrepareMarkStriation], Form(...)], mask_data: bytes = File(...)
) -> PrepareMarkResponseStriation:
    """Prepare the ScanFile, save it to the vault and return the URLs to access the files."""
    vault = create_vault(params.tag)
    parsed_image = parse_scan_pipeline(params.scan_file, 1, 1)

    try:
        parsed_mask = parse_mask_pipeline(
            raw_data=mask_data,
            shape=parsed_image.data.shape,
            is_bitpacked=params.mask_is_bitpacked,
        )
    except ArrayShapeMismatchError as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e))

    process_prepare_striation_mark(
        working_dir=vault.resource_path,
        scan_image=parsed_image,
        mark_type=params.mark_type,
        mask=parsed_mask,
        bounding_box=params.bounding_box,
        preprocess_parameters=params.mark_parameters,
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseStriation.from_enum(enum=PrepareMarkStriationFiles, base_url=vault.access_url)


@preprocessor_route.post(
    path=f"/{PreprocessorEndpoint.EDIT_SCAN}",
    summary="Validate and parse a scan file with edit parameters.",
    description="""
    Parse and validate a scan file (X3P format only) with the provided edit parameters
    (mask, crop, subsampling). Creates a new vault for storing future outputs.

    The mask shape specified in `mask_parameters.shape` must exactly match the shape
    (height × width) of the parsed scan image.

    Note: Image generation is currently not implemented.
""",
    responses={
        HTTPStatus.BAD_REQUEST: {"description": "parse error"},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"description": "mask shape does not match image shape"},
        HTTPStatus.INTERNAL_SERVER_ERROR: {
            "description": "processing error",
        },
    },
    openapi_extra=_generate_openapi_schema(model=EditImage),
)
async def edit_scan(params: Annotated[Json[EditImage], Form(...)], mask_data: bytes = File(...)) -> GeneratedImages:
    """
    Validate and parse a scan file with edit parameters and mask.

    Accepts an X3P scan file and edit parameters (mask, zoom, step sizes),
    validates the file format, parses it according to the parameters, and
    creates a vault directory for future outputs. Returns access URLs for the vault.
    """
    vault = create_vault(params.tag)
    logger.debug(f"Working directory created on: {vault.resource_path}")
    parsed_image = parse_scan_pipeline(params.scan_file, 1, 1)

    try:
        parsed_mask = parse_mask_pipeline(
            raw_data=mask_data,
            shape=parsed_image.data.shape,
            is_bitpacked=params.mask_is_bitpacked,
        )

    except ArrayShapeMismatchError as e:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(e))

    edited_scan_image = edit_scan_image(scan_image=parsed_image, edit_image_params=params, mask=parsed_mask)
    preview_pipeline(
        parsed_scan=edited_scan_image, output_path=GeneratedImageFiles.preview_image.get_file_path(vault.resource_path)
    )
    surface_map_pipeline(
        parsed_scan=edited_scan_image,
        output_path=GeneratedImageFiles.surface_map_image.get_file_path(vault.resource_path),
        light_sources=LIGHT_SOURCES,
        observer=OBSERVER,
    )
    logger.info(f"Generated files saved to {vault}")
    return GeneratedImages.from_enum(enum=GeneratedImageFiles, base_url=vault.access_url)
