from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import RedirectResponse
from loguru import logger
from pydantic import Json

from constants import (
    LIGHT_SOURCES,
    OBSERVER,
    PreprocessorEndpoint,
    RoutePrefix,
)
from extractors import ProcessedDataAccess
from extractors.constants import (
    GeneratedImageFiles,
    PrepareMarkImpressionFiles,
    PrepareMarkStriationFiles,
    ProcessFiles,
)
from extractors.schemas import (
    GeneratedImages,
    PrepareMarkResponseImpression,
    PrepareMarkResponseStriation,
    generate_model_with_urls,
)
from file_services import create_vault
from preprocessors.controller import edit_scan_image, process_prepare_impression_mark, process_prepare_striation_mark

from .pipelines import (
    parse_mask_pipeline,
    parse_scan_pipeline,
    preview_pipeline,
    surface_map_pipeline,
    x3p_pipeline,
)
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
    return generate_model_with_urls("ProcessedDataAccess", ProcessFiles, vault.access_url)


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
    process_prepare_impression_mark(
        scan_file=prepare_mark_parameters.scan_file,
        mark_type=prepare_mark_parameters.mark_type,
        mask=prepare_mark_parameters.mask_array,
        bounding_box=prepare_mark_parameters.bounding_box,
        preprocess_parameters=prepare_mark_parameters.mark_parameters,
        working_dir=vault.resource_path,
    )
    logger.info(f"Generated files saved to {vault}")
    return generate_model_with_urls("PrepareMarkResponseImpression", PrepareMarkImpressionFiles, vault.access_url)


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
    process_prepare_striation_mark(
        working_dir=vault.resource_path,
        scan_file=prepare_mark_parameters.scan_file,
        mark_type=prepare_mark_parameters.mark_type,
        mask=prepare_mark_parameters.mask_array,
        bounding_box=prepare_mark_parameters.bounding_box,
        preprocess_parameters=prepare_mark_parameters.mark_parameters,
    )
    logger.info(f"Generated files saved to {vault}")
    return generate_model_with_urls("PrepareMarkResponseStriation", PrepareMarkStriationFiles, vault.access_url)


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
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "properties": {
                            "params": EditImage.model_json_schema(),
                            "mask_data": {"type": "string", "format": "binary", "example": b"\x01\x00\x00\x01"},
                        },
                        "required": ["params", "mask_data"],
                    }
                },
                "application/json": {
                    "schema": {
                        "properties": {
                            "params": EditImage.model_json_schema(),
                        },
                        "required": ["params"],
                    }
                },
            }
        }
    },
)
async def edit_scan(
    params: Annotated[Json[EditImage], Form(...)], mask_data: Annotated[UploadFile, File(...)]
) -> GeneratedImages:
    """
    Validate and parse a scan file with edit parameters and mask.

    Accepts an X3P scan file and edit parameters (mask, zoom, step sizes),
    validates the file format, parses it according to the parameters, and
    creates a vault directory for future outputs. Returns access URLs for the vault.
    """
    vault = create_vault(params.tag)
    logger.debug(f"Working directory created on: {vault.resource_path}")
    parsed_image = parse_scan_pipeline(params.scan_file, 1, 1)
    parsed_mask = parse_mask_pipeline(
        raw_data=await mask_data.read(),
        shape=params.mask_parameters.shape,
        is_bitpacked=params.mask_parameters.is_bitpacked,
    )

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
    return generate_model_with_urls("GeneratedImages", GeneratedImageFiles, vault.access_url)
