from enum import StrEnum, auto
from typing import Annotated

from container_models.light_source import LightSource
from pydantic import AfterValidator, Field, FilePath, HttpUrl, field_validator

from models import BaseModelConfig


class SupportedPostExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


class SupportedGetExtension(StrEnum):
    X3P = auto()
    PNG = auto()


def validate_file_extension(file_name: str) -> str:
    """Validate that the file has a supported extension."""
    if not file_name.endswith(tuple(SupportedGetExtension)):
        raise ValueError(f"File must have one of these extensions: {', '.join(SupportedGetExtension)}")
    return file_name


type FileName = Annotated[
    str,
    AfterValidator(validate_file_extension),
    Field(
        ...,
        description=f"Filename of type: {','.join(SupportedGetExtension)}",
        examples=["example.png", "scan.x3p"],
    ),
]


class UploadScanParameters(BaseModelConfig):
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
    parameters: UploadScanParameters = Field(
        default_factory=UploadScanParameters.model_construct,
    )

    @property
    def name(self) -> str:
        return self.scan_file.stem

    @property
    def surfacemap_filename(self) -> str:
        return f"{self.name}_surfacemap.png"

    @property
    def preview_filename(self) -> str:
        return f"{self.name}_preview.png"

    @property
    def x3p_filename(self) -> str:
        return f"{self.name}.x3p"

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is of a supported type and not empty."""
        if scan_file.suffix[1:] not in SupportedPostExtension:
            raise ValueError(f"unsupported extension: {scan_file.name}")

        if scan_file.stat().st_size == 0:
            raise ValueError(f"file is empty: {scan_file.name}")

        return scan_file


class ProcessedDataLocation(BaseModelConfig):
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
