from __future__ import annotations

from enum import StrEnum, auto
from functools import cached_property
from typing import Annotated, Self, override

import numpy as np
from container_models.light_source import LightSource
from numpy.typing import NDArray
from pydantic import (
    UUID4,
    ConfigDict,
    EncodedBytes,
    EncoderProtocol,
    Field,
    FilePath,
    HttpUrl,
    PositiveFloat,
    RootModel,
    field_validator,
    model_validator,
)

from models import BaseModelConfig, ProjectTag, ScanFile, SupportedScanExtension


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
    scale_x: float = Field(
        1.0,
        gt=0.0,
        description="Horizontal pixel size in meters (m). Defines physical spacing between pixels in x-direction.",
    )
    scale_y: float = Field(
        1.0,
        gt=0.0,
        description="Vertical pixel size in meters (m). Defines physical spacing between pixels in y-direction.",
    )
    step_size_x: int = Field(
        1, gt=0, description="Subsampling step in x-direction. Values > 1 reduce resolution by skipping pixels."
    )
    step_size_y: int = Field(
        1, gt=0, description="Subsampling step in y-direction. Values > 1 reduce resolution by skipping pixels."
    )


class UploadScan(BaseModelConfig):
    """Request model for uploading and processing scan files."""

    scan_file: ScanFile
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


class MaskEncoder(EncoderProtocol):
    @override
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        if not data:
            raise ValueError("Cannot decode empty bytes to mask array")
        # Strict bool evaluation
        if any(byte_ not in (1, 0) for byte_ in data):
            raise ValueError("Corrupted encoding")
        return data

    @override
    @classmethod
    def encode(cls, value: bytes) -> bytes:
        # Ensure output is strict bool bytes
        return np.frombuffer(value, dtype=np.int8).astype(np.bool_).tobytes()

    @classmethod
    @override
    def get_json_format(cls) -> str:
        return "mask-bytes"


class Mask(BaseModelConfig):
    data: Annotated[bytes, EncodedBytes(MaskEncoder)]
    shape: tuple[int, int]

    @cached_property
    def mask_array(self) -> NDArray:
        """TODO: Add docstring."""
        return np.frombuffer(self.data, dtype=np.uint8).reshape(*self.shape).astype(np.bool_)

    @model_validator(mode="after")
    def validate_mask_model(self) -> Self:
        """TODO: Add docstring."""
        try:
            self.mask_array
        except Exception as e:
            raise ValueError("Fail to decode mask data") from e
        if self.mask_array.ndim != 2:  # noqa
            raise ValueError(f"Mask must be a 2D array: {self.mask_array.ndim} dimensions found")
        return self


class EditImageParameters(BaseModelConfig):
    """Configuration parameters for scan image editing and transformation operations."""

    mask: Mask = Field(
        description=(
            "Binary mask array for selective processing or cropping. "
            "Must be a 2D boolean array with shape matching the scan image dimensions. "
            "Regions marked true (1) are processed, false (0) regions are excluded. "
        ),
        examples=[
            [[True, True, False], [False, True, True]],
            [[1, 1, 0], [0, 1, 1]],
        ],
    )
    cutoff_length: PositiveFloat = Field(
        description="Cutoff wavelength in meters (m) for Gaussian regression filtering. "
        "Defines the spatial frequency threshold for surface texture analysis."
    )
    resampling_factor: PositiveFloat = Field(
        default=4,
        description="Resampling rate for image resolution adjustment. Higher values increase resolution.",
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
    step_size_x: int = Field(
        1,
        gt=0,
        description="Subsampling step size in x-direction. Values > 1 reduce resolution by skipping pixels.",
    )
    step_size_y: int = Field(
        1,
        gt=0,
        description="Subsampling step size in y-direction. Values > 1 reduce resolution by skipping pixels.",
    )
    overwrite: bool = Field(
        False,
        description="Whether to overwrite the original scan file with edited results.",
    )


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
        """Validate given file is off a supported type."""
        if scan_file.suffix[1:] != SupportedScanExtension.X3P:
            raise ValueError(f"unsupported extension: {scan_file.name}")
        return scan_file


class ProcessDataUrls(RootModel):
    """
    Collection of HTTP URLs pointing to processed scan data files.

    Provides immutable tuple of URLs for accessing generated outputs
    such as X3P files, surface maps, and preview images.
    """

    root: tuple[HttpUrl, ...] = Field(
        ...,
        description="Tuple of HTTP URLs for accessing processed scan files (X3P, surface maps, preview images).",
        examples=[
            (
                "http://localhost:8000/extractor/files/a1b2c3d4/project/scan.x3p",
                "http://localhost:8000/extractor/files/a1b2c3d4/project/surface_map.png",
                "http://localhost:8000/extractor/files/a1b2c3d4/project/preview.png",
            ),
        ],
    )
    model_config = ConfigDict(frozen=True, regex_engine="rust-regex")

    @model_validator(mode="after")
    def validate_non_empty(self) -> ProcessDataUrls:
        """Validate that the URL collection contains at least one URL."""
        if not self.root:
            raise ValueError("ProcessDataUrls must contain at least 1 item, received empty tuple")
        return self


class ProcessScanOutput(BaseModelConfig):
    """
    Response model for scan upload and processing operations.

    Contains URLs for accessing processed scan outputs and links to
    subsequent editing operations.
    """

    downloads: ProcessDataUrls = Field(
        ...,
        description=(
            "Collection of HTTP URLs for downloading processed scan files "
            "including X3P format, surface map visualization, and preview image."
        ),
    )
    token: UUID4
