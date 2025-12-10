from pydantic import BaseModel
from pydantic.fields import Field


class ImageGenerationError(BaseModel):
    message: str = Field(
        default="failed to generate preview_image.png",
        examples=[
            "Failed to export scan_image.x3p",
            "Failed to export preview_image.png",
            "Failed to export surface_map.x3p",
        ],
    )
    error: str = Field(default="ValueError: ....", examples=["ValueError: ...."])


class ParsingError(BaseModel):
    message: str = Field(default="failed to parse scan file", examples=["Failed to parse scan file"])
    error: str = Field(default="ExportError: ....", examples=["FileIsCorrupted: ...."])
