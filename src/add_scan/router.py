from fastapi import APIRouter


class ParseError(Exception): ...


add_scan = APIRouter(
    prefix="/add-scan",
    tags=["add-scan"],
)


@add_scan.post(
    path="/",
    summary="Add a scan file to be processed",
    description="""""",
)
async def add_scan_root() -> dict[str, str]:
    """TODO."""
    # parse parse incoming file
    # raise Unable to ParseError
    # subsample the parsed file
    # export newly created files to output directory
    # create surface map png
    # export png to output directory
    return {"message": "Hello from add-scan"}
