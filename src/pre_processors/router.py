from fastapi import APIRouter
from parsers import ScanImage

from .schemas import ProcessScan, UploadScan


class ParseError(Exception): ...


pre_processors = APIRouter(
    prefix="/pre-processor",
    tags=["pre-processor"],
)


@pre_processors.get(
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


@pre_processors.post(
    path="/process-scan",
    summary="Add a scan file to be processed",
    description="""
    Processes an uploaded scan file and generates several derived outputs, including
    an X3P file, a preview image, and a surface map, these files are saved to the working directory given as parameter.
    The endpoint parses and validates the file before running the processing pipeline.
""",
    responses={
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "examples": {
                        "path_not_exists": {
                            "summary": "Path doesn't exist",
                            "value": {"error": "path doesn't exist"},
                        },
                        "unsupported_extension": {
                            "summary": "Unsupported extension",
                            "value": {"error": "unsupported extension"},
                        },
                        "file_corrupt": {
                            "summary": "File is corrupt",
                            "value": {"error": "file is corrupt, file can't be parsed."},
                        },
                    }
                }
            },
        },
    },
)
async def process_scan(upload_scan: UploadScan) -> ProcessScan:
    """
    Process an uploaded scan file and generate derived output files.

    This endpoint parses and validates the incoming scan file, performs the
    necessary processing steps, and produces several outputs such as an X3P
    file, a preview image, and a surface map saved to the working directory.
    """
    # parse parse incoming file
    _ = ScanImage.from_file(upload_scan.scan_file)
    # raise Unable to ParseError
    # subsample the parsed file
    # export newly created files to output directory
    # create surface map png
    # export png to output directory
    # TODO: replace the arguments with actual calculated results
    return ProcessScan(
        x3p_image=upload_scan.output_dir / "circle.x3p",
        preview_image=upload_scan.output_dir / "preview.png",
        surfacemap_image=upload_scan.output_dir / "surfacemap.png",
    )
