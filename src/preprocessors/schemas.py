from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from pydantic import Field, FilePath, HttpUrl, RootModel, field_validator, model_validator

from models import BaseModelConfig


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class RenderedImage(
    RootModel,
):
    root: FilePath = Field(
        ..., description="parsed scan file.", examples=[Path("./temp/surface_map.png"), Path("./temp/preview.png")]
    )

    @field_validator("root", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] != "png":
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class UploadScan(BaseModelConfig):
    scan_file: FilePath = Field(
        ...,
        description="Upload scan file.",
        examples=[Path("./temp/scan.al3d"), Path("./temp/scan.x3p"), Path("./temp/scan.sur"), Path("./temp/scan.plu")],
    )

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] not in SupportedExtension:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class EditImageOptions(BaseModelConfig):
    scan_file: FilePath = Field(
        ...,
        description="Upload scan file.",
        examples=[Path("./temp/scan.al3d"), Path("./temp/scan.x3p"), Path("./temp/scan.sur"), Path("./temp/scan.plu")],
    )
    mask: Any | None = Field(None, description="Mask to mask the image or crop the image in combination with zoom.")
    zoom: bool = Field(
        False, description="Whether to zoom into the image based on the masked fields(can be from earlier eddits)."
    )

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] != "x3p":
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file

    @model_validator(mode="after")
    def validate_edits_are_chosen(self) -> "EditImageOptions":
        """Validate at least one edit option is chosen."""
        if self.mask is None and not self.zoom:
            raise ValueError("At least one  zoom must be provided.")
        return self


class ProcessedDataAndScanLocation(BaseModelConfig):
    x3p_image: HttpUrl = Field(
        ...,
        description="converted subsampled X3P image.",
        examples=["http://localhost:8000/preprocessor/file/surface_comparator_859lquto/scan.x3p"],
    )
    preview_image: HttpUrl = Field(
        ...,
        description="rgba image made from the x3p converted file.",
        examples=["http://localhost:8000/preprocessor/file/surface_comparator_859lquto/preview.png"],
    )
    surfacemap_image: HttpUrl = Field(
        ...,
        description="surface image made from the x3p converted file.",
        examples=["http://localhost:8000/preprocessor/file/surface_comparator_859lquto/surface_map.png"],
    )


class ProcessedDataLocation(BaseModelConfig):
    preview_image: HttpUrl = Field(
        ...,
        description="rgba image made from the x3p converted file.",
        examples=["http://localhost:8000/preprocessor/image_file/surface_comparator_859lquto/preview.png"],
    )
    surfacemap_image: HttpUrl = Field(
        ...,
        description="surface image made from the x3p converted file.",
        examples=["http://localhost:8000/preprocessor/image_file/surface_comparator_859lquto/surface_map.png"],
    )
