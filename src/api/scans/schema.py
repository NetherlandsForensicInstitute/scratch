from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Scan(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the table/schema",
        examples=["fef500b88b8f4d62be1f8a2429a964e7", "45b6cfd9327342a0bcaba6c8ab20a7ca"],
    )
    type: Literal["Huls", "Bullet"] = Field(
        ..., description="The type of scan, this can be 'Bullet' or 'Huls'", examples=["Huls", "Bullet"]
    )
    scanned_file: Path = Field(..., description="Path to the 3D scanned file", examples=[Path("./ding/somthing.stl")])
    heatmap: Path | None = Field(
        None,
        description="""Path to the converted heatmap,when not converted value is set to None""",
        examples=[None, Path("./ding/somthing.png")],
    )
    surface_map: Path | None = Field(
        None,
        description="""Path to the converted surface_map,when not converted value is set to None""",
        examples=[None, Path("./ding/somthing.png")],
    )
    edited_map: Path | None = Field(
        None,
        description="Path to the edited map of the surface map edited from the surface map",
        examples=[None, Path("./ding/somthing.png")],
    )
