from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from image_generation.image_generation import generate_3d_image
from loguru import logger
from parsers import from_file
from parsers.exceptions import ExportError
from parsers.x3p import save_to_x3p

from preprocessors.models import ImageGenerationError, ParsingError

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
    try:
        parsed_scan = from_file(upload_scan.scan_file).subsample_data(step_x=1, step_y=1)
    except ExportError as err:
        logger.error("jammer man, failed to parse the given scan file")
        raise HTTPException(status_code=400, detail=f"Failed to parse the given scan file, err:{str(err)}")
    try:
        save_to_x3p(image=parsed_scan, output_path=upload_scan.output_dir / "scan.x3p")
    except ExportError as err:
        logger.error("jammer man, failed to save the scan file")
        raise HTTPException(
            status_code=500, detail=f"Failed to save the scan file  {upload_scan.output_dir / 'scan.x3p'}: {str(err)}"
        )

    for image_generator, file_name in zip([generate_3d_image, generate_3d_image], ["surface_map.png", "preview.png"]):
        try:
            image_generator(parsed_scan).image.save(
                upload_scan.output_dir / file_name
            )  # TODO: if we want somthing like this, protocol is needed
        except ValueError as err:  # TODO: ugly but it is how it is now..
            logger.error("jammer man, failed to parse the given scan file")
            raise HTTPException(status_code=500, detail=f"Failed to generate {file_name}: {str(err)}")

    return ProcessedDataLocation(
        x3p_image=upload_scan.output_dir / "scan.x3p",
        preview_image=upload_scan.output_dir / "preview.png",
        surfacemap_image=upload_scan.output_dir / "surface_map.png",
    )
