from fastapi import APIRouter
from parsers import load_scan_image

from .schemas import EditImage, ProcessedDataLocation, UploadScan


class ParseError(Exception): ...


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
)
async def process_scan(upload_scan: UploadScan) -> ProcessedDataLocation:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the output directory.
    """
    # parse incoming file
    _ = load_scan_image(upload_scan.scan_file)
    # raise Unable to ParseError
    # subsample the parsed file
    # export newly created files to output directory
    # create surface map png
    # export png to output directory
    # TODO: replace the arguments with actual calculated results
    return ProcessedDataLocation(
        x3p_image=upload_scan.output_dir / "circle.x3p",
        preview_image=upload_scan.output_dir / "preview.png",
        surfacemap_image=upload_scan.output_dir / "surface_map.png",
    )


@preprocessor_route.post(
    path="/edit-image",
    summary="Edit parsed scan",
    description="""""",
)
async def edit_image(params: EditImage) -> dict[str, str]:
    """TODO."""
    # load requested x3p file
    # create an edited preview image (png)
    # create surface maps (png)
    # response contains path to x3p file and two pngs (preview and surface_map)
    return {"message": "Hello from edit-image"}
