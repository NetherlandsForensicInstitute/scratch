from enum import StrEnum, auto
from typing import Annotated

from container_models.light_source import LightSource
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from pydantic import AfterValidator, Field, FilePath, field_validator

from constants import ImpressionMarks, MaskTypes, StriationMarks
from models import BaseModelConfig, ProjectTag, validate_file_extension, validate_not_executable


class SupportedExtension(StrEnum):
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    PLU = auto()


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
        """Get model fields as dict with optional filtering."""
        return {field: getattr(self, field) for field in self.model_dump(exclude=exclude, include=include)}


class BaseParameters(BaseModelConfig):
    """Base parameters for preprocessor operations including scan file."""

    project_name: ProjectTag | None = Field(None, description="", examples=[])
    scan_file: Annotated[
        FilePath,
        AfterValidator(lambda filepath: validate_file_extension(filepath, SupportedExtension)),
        AfterValidator(validate_not_executable),
    ] = Field(
        ...,
        description=f"Path to the input scan file. Supported formats: {', '.join(SupportedExtension)}",
    )

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return self.project_name or self.scan_file.stem


class UploadScan(BaseParameters):
    parameters: UploadScanParameters = Field(default_factory=UploadScanParameters.model_construct)

    @field_validator("scan_file", mode="after")
    @classmethod
    def _validate_scan_file(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is not empty."""
        if scan_file.stat().st_size == 0:
            raise ValueError(f"file is empty: {scan_file.name}")

        return scan_file


class CropInfo(BaseModelConfig):
    type: MaskTypes
    data: dict
    is_foreground: bool


class PreprocessingStriationParams(BaseModelConfig):
    """dummy till #84 is merged."""

    pass  # TODO: not yet merged dataclass from PR #84


class PrepareMarkStriation(BaseParameters):
    mark_type: StriationMarks = Field(..., description="Type of mark to prepare.")
    mask_array: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    rotation_angle: int = Field(0, description="Rotation angle for the mark preparation.")
    crop_info: CropInfo | None = Field(
        None, description="", examples=[{"type": "rectangle", "data": {}, "is_foreground": False}]
    )
    mark_parameters: PreprocessingStriationParams = Field(
        ..., description="Preprocessor parameters."
    )  # TODO: not yet merged dataclass from PR #84


class PrepareMarkImpression(BaseParameters):
    mark_type: ImpressionMarks = Field(..., description="Type of mark to prepare.")
    mask_array: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    rotation_angle: int = Field(0, description="Rotation angle for the mark preparation.")
    crop_info: CropInfo | None = Field(
        None, description="", examples=[{"type": "rectangle", "data": {}, "is_foreground": False}]
    )
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")
