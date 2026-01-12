from __future__ import annotations

from typing import Any

from container_models.light_source import LightSource
from pydantic import ConfigDict, Field, FilePath, HttpUrl, RootModel, field_validator, model_validator

from models import BaseModelConfig, ParametersModel, ProjectTag, ScanFile, SupportedScanExtension


class UploadScanParameters(ParametersModel):
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


class UploadScan(BaseModelConfig):
    scan_file: ScanFile
    project_name: ProjectTag | None = Field(None, description="", examples=[])
    parameters: UploadScanParameters = Field(default_factory=UploadScanParameters.model_construct)

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return self.project_name or self.scan_file.stem

    @field_validator("scan_file", mode="after")
    @classmethod
    def _validate_scan_file(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is not empty."""
        if scan_file.stat().st_size == 0:
            raise ValueError(f"file is empty: {scan_file.name}")

        return scan_file


class EditImageParameters(ParametersModel):
    mask: Any | None = Field(None, description="Mask to mask the image or crop the image in combination with zoom.")
    zoom: bool = Field(
        False, description="Whether to zoom into the image based on the masked fields(can be from earlier eddits)."
    )
    step_size_x: int = Field(1, gt=0)
    step_size_y: int = Field(1, gt=0)
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_edits_are_chosen(self) -> EditImageParameters:
        """Validate at least one edit option is chosen."""
        if self.mask is None and not self.zoom:
            raise ValueError("At least one  zoom must be provided.")
        return self


class EditImage(BaseModelConfig):
    scan_file: ScanFile
    project_name: ProjectTag | None = Field(None, description="", examples=[])
    parameters: EditImageParameters = Field(default_factory=EditImageParameters.model_construct)

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return self.project_name or self.scan_file.stem

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] != SupportedScanExtension.X3P:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class ProcessDataUrls(RootModel):
    root: tuple[HttpUrl, ...]
    model_config = ConfigDict(frozen=True, regex_engine="rust-regex")


class ProcessScanOutput(BaseModelConfig):
    edit_scan: HttpUrl = Field(..., description="next endpoint")
    downloads: ProcessDataUrls = Field(..., description="")
