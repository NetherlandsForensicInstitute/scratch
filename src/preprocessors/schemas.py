from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from container_models.light_source import LightSource
from pydantic import DirectoryPath, Field, FilePath, field_validator, model_validator

from models import BaseModelConfig


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class UploudScanParameters(BaseModelConfig):
    """Configuration parameters for upload scan's surface rendering process."""

    light_sources: tuple[LightSource, ...] = Field(
        (
            LightSource(azimuth=90, elevation=45),
            LightSource(azimuth=180, elevation=45),
        ),
        description="Light sources for surface illumination rendering.",
    )
    observer: LightSource = Field(
        LightSource(azimuth=90, elevation=45),
        description="Observer viewpoint vector for surface rendering.",
    )
    scale_x: float = Field(1.0, gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(1.0, gt=0.0, description="pixel size in meters (m)")
    step_size_x: int = Field(1, gt=0)
    step_size_y: int = Field(1, gt=0)

    def as_dict(self, *, exclude: set[str] | None = None, include: set[str] | None = None) -> dict:
        """
        Get model fields as dict with nested models intact (not serialized).

        :param exclude: Set of field names to exclude
        :param include: Set of field names to include (mutually exclusive with exclude)
        """
        if exclude and include:
            raise ValueError("Cannot specify both 'exclude' and 'include'")

        fields = set(self.__class__.model_fields)

        if include:
            fields = include
        elif exclude:
            fields = fields - exclude

        return {field: getattr(self, field) for field in fields}


class UploadScan(BaseModelConfig):
    scan_file: FilePath = Field(
        ...,
        description="Path to the input scan file. Supported formats: AL3D, X3P, SUR, PLU.",
    )
    output_dir: DirectoryPath = Field(
        ...,
        description="Directory where processed outputs (X3P, preview, and surface map images) will be saved.",
    )
    parameters: UploudScanParameters = Field(
        default_factory=UploudScanParameters.model_construct,
    )

    @property
    def surfacemap_path(self) -> Path:
        return self.__output_partial_path("_surfacemap").with_suffix(".png")

    @property
    def preview_path(self) -> Path:
        return self.__output_partial_path("_preview").with_suffix(".png")

    @property
    def x3p_path(self) -> Path:
        return self.__output_partial_path().with_suffix(".x3p")

    def __output_partial_path(self, postfix: str | None = None) -> Path:
        return self.output_dir / f"{self.scan_file.stem}{postfix or ''}"

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is of a supported type and not empty."""
        if scan_file.suffix[1:] not in SupportedExtension:
            raise ValueError(f"unsupported extension: {scan_file.name}")

        if scan_file.stat().st_size == 0:
            raise ValueError(f"file is empty: {scan_file.name}")

        return scan_file


class ProcessedDataLocation(BaseModelConfig):
    x3p_image: FilePath = Field(
        ..., description="converted subsampled X3P image.", examples=[Path("./documents/project_x/x3p.png")]
    )
    preview_image: FilePath = Field(
        ...,
        description="rgba image made from the x3p converted file.",
        examples=[Path("./documents/project_x/preview.png")],
    )
    surfacemap_image: FilePath = Field(
        ...,
        description="surface image made from the x3p converted file.",
        examples=[Path("./documents/project_x/surfacemap.png")],
    )

    @model_validator(mode="after")
    def same_parent_directory(self) -> Self:
        """Validate that all files are in the same parent directory."""
        if not all(
            getattr(self, field_name).parent == self.x3p_image.parent
            for field_name, field_info in self.__class__.model_fields.items()
            if field_info.annotation is Path
        ):
            raise ValueError("All fields must point to the same output directory")
        return self
