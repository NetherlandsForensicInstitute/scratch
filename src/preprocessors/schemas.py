from __future__ import annotations

from enum import StrEnum, auto
from functools import cached_property
from typing import Annotated, Self

import numpy as np
from container_models.light_source import LightSource
from numpy.typing import NDArray
from pydantic import (
    AfterValidator,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from scipy.constants import micro

from models import BaseModelConfig, ProjectTag, ScanFile, SupportedScanExtension


class UploadScanParameters(BaseModelConfig):
    """Configuration parameters for upload scan's surface rendering process."""

    light_sources: tuple[LightSource, ...] = Field(
        (
            LightSource(azimuth=90, elevation=45),
            LightSource(azimuth=180, elevation=45),
        ),
        description="Light sources for surface illumination rendering.",
        examples=[
            (
                LightSource(azimuth=90, elevation=45),
                LightSource(azimuth=180, elevation=45),
            ),
        ],
    )
    observer: LightSource = Field(
        LightSource(azimuth=90, elevation=45),
        description="Observer viewpoint vector for surface rendering.",
        examples=[LightSource(azimuth=90, elevation=45)],
    )
    scale_x: float = Field(
        1.0,
        gt=0.0,
        description="Horizontal pixel size in meters (m). Defines physical spacing between pixels in x-direction.",
        examples=[1.0, 0.5, 2.0],
    )
    scale_y: float = Field(
        1.0,
        gt=0.0,
        description="Vertical pixel size in meters (m). Defines physical spacing between pixels in y-direction.",
        examples=[1.0, 0.5, 2.0],
    )
    step_size_x: int = Field(
        1,
        gt=0,
        description="Subsampling step in x-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )
    step_size_y: int = Field(
        1,
        gt=0,
        description="Subsampling step in y-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )


class UploadScan(BaseModelConfig):
    """Request model for uploading and processing scan files."""

    scan_file: ScanFile = Field(
        ...,
        description=f"Path to the input scan file. Supported formats: {', '.join(SupportedScanExtension)}",
    )
    project_name: ProjectTag | None = Field(
        None,
        description=(
            "Optional project identifier for organizing uploaded scans. "
            "Used as directory tag if provided, otherwise defaults to scan filename."
        ),
        examples=["forensic_analysis_2026", "case_12345"],
    )
    parameters: UploadScanParameters = Field(
        default_factory=UploadScanParameters.model_construct,
        description=(
            "Configuration parameters controlling scan processing, "
            "including lighting, scaling, and subsampling settings."
        ),
    )

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


class Terms(StrEnum):
    """Surface fitting terms for leveling operations."""

    PLANE = auto()
    SPHERE = auto()


class RegressionOrder(StrEnum):
    """Polynomial regression order for surface leveling."""

    RO = auto()
    R1 = auto()
    R2 = auto()


type Mask = tuple[tuple[bool, ...], ...]


class EditImageParameters(BaseModelConfig):
    """Configuration parameters for scan image editing and transformation operations."""

    mask: Mask = Field(
        description=(
            "Binary mask defining regions to include (True/1) or exclude (False/0) during processing. "
            "Accepts both boolean (True/False) and integer (1/0) representations. "
            "Must be a 2D tuple structure matching or smaller than the scan dimensions."
        ),
        examples=[
            ((1, 0), (0, 1)),  # Integer format
            ((True, False), (False, True)),  # Boolean format
        ],
    )
    cutoff_length: Annotated[PositiveFloat, AfterValidator(lambda x: x * micro)] = Field(
        description="Cutoff wavelength in micrometers (µm) for Gaussian regression filtering. "
        "Defines the spatial frequency threshold for surface texture analysis.",
        examples=[250, 500, 1000],
    )
    resampling_factor: PositiveFloat = Field(
        default=4,
        description="Resampling rate for image resolution adjustment. Higher values increase resolution.",
        examples=[2, 4, 8],
    )
    terms: Terms = Field(
        default=Terms.PLANE,
        description=(
            "Surface fitting model for leveling operations. PLANE for planar surfaces, SPHERE for curved surfaces."
        ),
    )
    regression_order: RegressionOrder = Field(
        default=RegressionOrder.RO,
        description="Polynomial regression order for surface fitting. R0 (constant), R1 (linear), or R2 (quadratic).",
    )
    crop: bool = Field(
        default=False,
        description="Whether to crop the image to the non-masked region.",
    )
    step_size_x: PositiveInt = Field(
        1,
        description="Subsampling step size in x-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )
    step_size_y: PositiveInt = Field(
        1,
        description="Subsampling step size in y-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )

    @model_validator(mode="after")
    def validate_mask_is_2d(self) -> Self:
        """
        Validate that the mask is a valid 2D array structure.

        Ensures the mask can be converted to a numpy array and has exactly
        2 dimensions, as required for image masking operations.
        """
        try:
            self.mask_array
        except (ValueError, TypeError) as e:
            raise ValueError("Bad mask value: unable to capture mask") from e
        if not self.mask_array.ndim == 2:  # noqa: PLR2004
            raise ValueError(f"Mask is not a 2D image: D{self.mask_array.ndim}")
        return self

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)


class EditImage(BaseModelConfig):
    """Request model for editing and transforming processed scan images."""

    scan_file: ScanFile
    project_name: ProjectTag | None = Field(
        None,
        description=(
            "Optional project identifier for organizing edited scans. "
            "Used as directory tag if provided, otherwise defaults to scan filename."
        ),
        examples=["forensic_analysis_2026", "case_12345"],
    )
    parameters: EditImageParameters = Field(
        default_factory=EditImageParameters.model_construct,
        description=(
            "Configuration parameters controlling image editing operations "
            "including resampling, leveling, masking, and cropping."
        ),
    )

    @property
    def tag(self) -> str:
        """Get the tag to use for directory naming."""
        return self.project_name or self.scan_file.stem

    @field_validator("scan_file", mode="after")
    @classmethod
    def validate_file_extension(cls, scan_file: FilePath) -> FilePath:
        """Validate that the given file is of a supported type (X3P format only)."""
        if scan_file.suffix[1:] != SupportedScanExtension.X3P:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file
