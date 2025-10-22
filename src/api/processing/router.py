from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import Response
from loguru import logger

processing_router = APIRouter(
    prefix="/processing",
    tags=["processing"],
)


@processing_router.get(
    path="/add_gun",
    # TODO ik zag dit niet in de matlab scripts, maar het moet wel ergens als input, moet ik nog ff uiztoeken
    summary="add details for a gun",
    description="""Get surface map as a PNG.
The surfacemap is retrieved from the database and converted to a png.
404 or so returned when the surfacemap is not found.\n
## MatLab functions:
- GetImageForDisplay
""",
    responses={
        200: {"description": "returns a 'png' image of the selected surfacemap", "content": {"image/png": {}}},
        404: {"description": "scan could not be uploaded"},
    },
    response_class=Response,
)
async def add_gun(id: str):
    """Docstring"""
    ...


@processing_router.post(
    path="/proces_image_{id_key}_{id_to_compare}",
    summary="proces a comparrision",
    description="""proces a one to one or one to many comparrision.
The edited surface-map is retrieved from the db together with the 'Type' of the scanned image (bullet or a huls).
**There is now a split between stration and impression in the processing plus 1-1 or 1-many for the  comparison,
I dont see what is done in the process category why it should be seperated from comparison**\n
## MatLab functions:
### For processing
- ProcesBulletData
- ProcesNIST
### For comparison
- ProfileCorrelatorSingle (one to one)
- CompareDatasetNIST (one to one)
- CompareDatasetsMultiple ( one to many)
- CompareDatasetMultipleNist (one to many)
""",
    responses={
        200: {
            "description": "proces started, returns a key uuid for the database location",
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
        404: {"description": "process could not be started"},
    },
    response_class=Response,
)
async def proces_image(
    proces_type: Literal["Impression", "Stiation"],
    id_key: str,
    id_to_compare: str | None = None,
) -> str:
    """Docstring"""
    logger.info(f"lets start comparing {id_key} with {id_to_compare} with process type:{proces_type}.")
    logger.info("lets proces stuff. vroom vroom. ")
    return uuid4().hex


# TODO: at this point also the surface image and preview image is called, but don't know why this is needed.

# TODO: from here on the comparison is made. but no clue if the split is really neccesarry
