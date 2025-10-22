from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger

from .schema import Scan

scan_router = APIRouter(
    prefix="/3d_scan",
    tags=["3d scan"],
)


@scan_router.post(
    path="/add_scan",
    summary="Upload 3d scan of a bullet/Huls/Pistol to a DB (How is the scan named in NFI??)",
    description="""
for now we can move the file to a location and put the path in a DB.
after that we need to convert the uploaded scan to two items, these two conversions are done in the background:
    1: for the 'heatmap'
    2: for the 'surfacemap'
When done converting moves the file to 'a' location an copies the filepath to the db both in there own column.\n
## MatLab functions:
- ImportDATA
- SubsampleData
- SaveDataTox3PFile
- GetSurfacePlot (if we convert it right away)
- GetPreviewImageForCropping (if we convert it right away)(dubble check this)
    """,
    responses={
        200: {
            "description": "scan successfully uploaded, returns a key uuid for the database location",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "string",
                        "format": "uuid",
                        "example": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                    }
                }
            },
        },
        404: {"description": "scan could not be uploaded"},
    },
)
async def add_scan(file: UploadFile) -> str:
    """Docstring"""
    logger.info(f"got file {file.filename}")
    return uuid4().hex


@scan_router.get(
    path="/scan_overview_{}",
    summary="get status of scan, used probably for polling",
    description="""
get the status of the scans.
Due to scan conversions are in a que, not all items are done at the same time.
When a conversion is done the Db item is updated with the Path to the conversion.
    """,
)
async def get_scan_state(id_key: str) -> Scan:
    """Docstring"""
    return Scan(id=id_key, type="Huls", scanned_file=Path())


@scan_router.get(
    path="/surfacemap_{}",
    summary="get surface map as a PNG",
    description="""Get surface map as a PNG.
The surfacemap is retrieved from the database and converted to a png.
404 or so returned when the surfacemap is not found.\n
## MatLab functions:
- GetImageForDisplay
    """,  # TODO: check how this is done, just a png to send over or a file?
    responses={
        200: {"description": "returns a 'png' image of the selected surfacemap", "content": {"image/png": {}}},
        404: {"description": "scan could not be uploaded"},
    },
    response_class=Response,
)
async def get_surfacemap(id_key: str) -> Response:
    """Docstring"""
    if id_key == "NFI":
        logger.info("found key in db")
        image_content = Path(r"./src/api/img.png").read_bytes()
        return Response(content=image_content, media_type="image/png")

    raise HTTPException(status_code=404, detail="only key=NFI is in the database")


@scan_router.get(
    path="/heatmap_{}",
    summary="get heatmap map as a PNG",
    description="""get heatmap map as a PNG.
The heatmap is retrieved from the database and converted to a png.
404 or so returned when the heatmap is not found.\n
## MatLab functions:
- GetImageForDisplay
    """,  # TODO: check how this is done, just a png to send over or a file?
    responses={
        200: {"description": "returns a 'png' image of the selected heatmap", "content": {"image/png": {}}},
        404: {"description": "scan could not be uploaded"},
    },
    response_class=Response,
)
async def get_heatmap(id_key: str) -> Response:
    """Docstring"""
    if id_key == "NFI":
        logger.info("found key in db")
        image_content = Path(r"./src/api/img.png").read_bytes()
        return Response(content=image_content, media_type="image/png")
    raise HTTPException(status_code=404, detail="only key=NFI is in the database")


@scan_router.get(
    path="/edit_surface_{}",
    summary="get edited_surface map as a PNG",
    description="""get edited_surface map as a PNG.
The edited surface is retrieved from the database and converted to a png.
404 or so returned when the edited surface map is not found.\n
## MatLab functions:
- GetImageForDisplay
    """,  # TODO: check how this is done, just a png to send over or a file?
    responses={
        200: {"description": "returns a 'png' image of the selected edited_surface", "content": {"image/png": {}}},
        404: {"description": "scan could not be uploaded"},
    },
    response_class=Response,
)
async def get_edit_image(id_key: str) -> Response:
    """Docstring"""
    if id_key == "NFI":
        logger.info("found key in db")
        image_content = Path(r"./src/api/img.png").read_bytes()
        return Response(content=image_content, media_type="image/png")
    raise HTTPException(status_code=404, detail="only key=NFI is in the database")


@scan_router.post(
    path="/edit_image_{}",
    summary="save edited surface map",
    description="""save the edited_surface map
returns the edited surface map.\n
## MatLab functions:
- RotateCropImage
- GetImageForDisplay
    """,  # TODO: check how this is done, just a png to send over or a file?
    responses={
        200: {"description": "returns a 'png' image of the edited_surface", "content": {"image/png": {}}},
        404: {"description": "scan could not be uploaded"},
    },
    response_class=Response,
)
async def edit_image_(
    id_key: str, image: Path
) -> Path:  # TODO: check how this is uploaded, i guesse a list of coordinates of geometry
    """Docstring"""
    return Scan(id=id_key, type="Huls", scanned_file=Path(), edited_map=image).edited_map


# TODO: wat doet ResampleMarkTypeSpecific somthing with 'station,impression?
