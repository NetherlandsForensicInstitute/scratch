from functools import partial
from http import HTTPStatus

from fastapi import APIRouter
from loguru import logger

from constants import PREPROCESSOR_ROUTE
from extractors import ProcessedDataAccess
from extractors.schemas import PrepareMarkResponseImpression, PrepareMarkResponseStriation
from file_services import create_vault, get_files, get_urls
from preprocessors.controller import process_prepare_mark

from .pipelines import (
    impression_mark_pipeline,
    parse_scan_pipeline,
    preview_pipeline,
    striation_mark_pipeline,
    surface_map_pipeline,
    x3p_pipeline,
)
from .schemas import PrepareMarkImpression, PrepareMarkStriation, UploadScan

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
    parsed_scan = parse_scan_pipeline(upload_scan.scan_file, upload_scan.parameters)
    files = get_files(vault.resource_path, scan="scan.x3p", preview="preview.png", surface_map="surface_map.png")
    x3p_pipeline(parsed_scan, files["scan"])
    surface_map_pipeline(parsed_scan, files["surface_map"], upload_scan.parameters)
    preview_pipeline(parsed_scan, files["preview"])

    logger.info(f"Generated files saved to {vault}")
    return ProcessedDataAccess(**get_urls(vault.access_url, **{key: file_.name for key, file_ in files.items()}))


@preprocessor_route.post(
    path="/prepare-mark-impression",
    summary="Preprocess a scan into analysis-ready mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for impression marks.

    Outputs two processed mark representations (.npy data and .json
    metadata) saved to the vault, returning URLs for file access.
    """,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def prepare_mark_impression(prepare_mark_parameters: PrepareMarkImpression) -> PrepareMarkResponseImpression:
    """Prepare the ScanFile, save it to the vault and return the urls to acces the files."""
    vault = create_vault(prepare_mark_parameters.tag)
    files = get_files(
        vault.resource_path,
        scan="scan.x3p",
        preview="preview.png",
        surface_map="surface_map.png",
        mark_file="mark.mat",
        processed_file="processed.mat",
        leveled_file="levelled.mat",
    )
    files = process_prepare_mark(
        files=files,
        scan_file=prepare_mark_parameters.scan_file,
        marking_method=partial(impression_mark_pipeline, params=prepare_mark_parameters.mark_parameters),
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseImpression(
        **get_urls(vault.access_url, **{key: file_.name for key, file_ in files.items()})
    )


@preprocessor_route.post(
    path="/prepare-mark-striation",
    summary="Preprocess a scan into analysis-ready mark files.",
    description="""
    Applies user-defined masking and cropping to a scan, then performs
    mark-type-specific preprocessing (rotation, cropping, filtering) for striation marks.

    Outputs two processed mark representations (.npy data and .json
    metadata) saved to the vault, returning URLs for file access.
    """,
    responses={
        HTTPStatus.INTERNAL_SERVER_ERROR: {"description": "image generation error"},
    },
)
async def prepare_mark_striation(prepare_mark_parameters: PrepareMarkStriation) -> PrepareMarkResponseStriation:
    """Prepare the ScanFile, save it to the vault and return the urls to acces the files."""
    vault = create_vault(prepare_mark_parameters.tag)
    files = get_files(
        vault.resource_path,
        scan="scan.x3p",
        preview="preview.png",
        surface_map="surface_map.png",
        mark_file="mark.mat",
        processed_file="processed.mat",
        profile_file="profile.mat",
    )
    process_prepare_mark(
        files=files,
        scan_file=prepare_mark_parameters.scan_file,
        marking_method=partial(striation_mark_pipeline, params=prepare_mark_parameters.mark_parameters),
    )
    logger.info(f"Generated files saved to {vault}")
    return PrepareMarkResponseStriation(
        **get_urls(vault.access_url, **{key: file_.name for key, file_ in files.items()})
    )
