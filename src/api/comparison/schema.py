from typing import Literal

from pydantic import BaseModel, Field


class ProcessResultOneToOne(BaseModel):
    """comparison results of a 1-1 comparison."""

    id: str = Field(
        ...,
        description="Unique identifier for the table/schema",
        examples=["3e1f959170104b288bbdac93ac871dc9", "83cd3bebfbd04587ba06098fbfae1e14"],
    )
    type: Literal["Stiation", "Impression"] = Field(
        ...,
        description="type for the processing the bullet/huls \nImpression is meant for ... \nStiation is meant for ...",
        examples=["Stiation", "Impression"],
    )
    plotdata: str  # TODO: check what this is
    score: int = Field(
        ..., description="The score of the processed comparison 0-100 (%)", examples=[85, 60], ge=0, le=100
    )
    scan_item_id: str = Field(
        ..., description="The id of the scan item to compare", examples=["fef500b88b8f4d62be1f8a2429a964e7"]
    )
    scan_comparison_id: str = Field(
        ..., description="The id of the scan item to compare", examples=["45b6cfd9327342a0bcaba6c8ab20a7ca"]
    )


class ProcessResultOneToMany(BaseModel):
    """comparison results of a 1 to many comparison."""

    ...
