from __future__ import annotations

from container_models.light_source import LightSource
from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
)

from constants import ImpressionMarks, MaskTypes, StriationMarks
from models import (
    BaseModelConfig,
    ProjectTag,
    ScanFile,
    SupportedScanExtension,
)


class BaseParameters(BaseModelConfig):
    """Base parameters for preprocessor operations including scan file."""

    project_name: ProjectTag | None = Field(
        None,
        description=(
            "Optional project identifier for organizing edited scans. "
            "Used as directory tag if provided, otherwise defaults to scan filename."
        ),
        examples=["forensic_analysis_2026", "case_12345"],
    )
    scan_file: ScanFile = Field(
        ...,
        description=f"Path to the input scan file. Supported formats: {', '.join(SupportedScanExtension)}",
    )

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return self.project_name or self.scan_file.stem


class UploadScan(BaseParameters):
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
    scale_x: PositiveFloat = Field(1.0, description="pixel size in meters (m)")
    scale_y: PositiveFloat = Field(1.0, description="pixel size in meters (m)")
    step_size_x: PositiveInt = Field(1)
    step_size_y: PositiveInt = Field(1)


class CropInfo(BaseModelConfig):
    type: MaskTypes
    data: dict
    is_foreground: bool


class PreprocessingImpressionParams(BaseModelConfig):
    """dummy till #84 is merged."""

    pass  # TODO: not yet merged dataclass from PR #84


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
