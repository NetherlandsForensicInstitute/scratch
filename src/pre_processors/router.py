from fastapi import APIRouter
from parsers import ScanImage

from .schemas import UploadScan


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
    path="/add-scan",
    summary="Add a scan file to be processed",
    description="""""",
)
async def add_scan(upload_scan: UploadScan) -> dict[str, str]:
    """TODO."""
    # parse parse incoming file
    _ = ScanImage.from_file(upload_scan.scan_file)
    # raise Unable to ParseError
    # subsample the parsed file
    # export newly created files to output directory
    # create surface map png
    # export png to output directory
    return {"message": "Hello from add-scan"}
