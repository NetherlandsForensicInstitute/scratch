from __future__ import annotations

from functools import cached_property

import numpy as np
from conversion.data_formats import BoundingBox, MarkImpressionType, MarkStriationType, SurfaceTermsAnnotated
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_striation import PreprocessingStriationParams
from pydantic import (
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from utils.constants import RegressionOrder

from models import (
    BaseModelConfig,
    ProjectTag,
    ScanFile,
    SupportedScanExtension,
)
from schemas import URLContainer

PIXEL_SIZE_MIN = 1e-9
PIXEL_SIZE_MAX = 1e-3
STEP_SIZE_MIN = 1
STEP_SIZE_MAX = 100
CUTOFF_MIN = 1e-6
CUTOFF_MAX = 2.5e-4
RESAMPLING_FACTOR_MIN = 0.1
RESAMPLING_FACTOR_MAX = 100


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
    scale_x: PositiveFloat = Field(
        1e-6,
        description="Horizontal pixel size in meters (m). Defines physical spacing between pixels in x-direction.",
        examples=[1e-6, 3.5e-6, 5e-6],
    )
    scale_y: PositiveFloat = Field(
        1e-6,
        description="Vertical pixel size in meters (m). Defines physical spacing between pixels in y-direction.",
        examples=[1e-6, 3.5e-6, 5e-6],
    )
    step_size: PositiveInt = Field(
        1,
        description="Sets the sampling interval for both axes. "
        "Values > 1 downscale the image by skipping intermediate pixels.",
        examples=[1, 2, 4],
    )

    @field_validator("scale_x", "scale_y")
    @classmethod
    def check_plausible_pixel_size(cls, v: float) -> float:
        """Check that the pixel size has a reasonable value."""
        if not (PIXEL_SIZE_MIN <= v <= PIXEL_SIZE_MAX):
            raise ValueError(
                f"scale value {v} m is outside the plausible range for a pixel size "
                f"({PIXEL_SIZE_MIN} to {PIXEL_SIZE_MAX} m). Did you enter micrometers instead of meters? "
                f"E.g. 3.5 µm should be 3.5e-6, not 3.5."
            )
        return v

    @field_validator("step_size")
    @classmethod
    def check_step_size(cls, v: float) -> float:
        """Check that the step size has a reasonable value."""
        if not (STEP_SIZE_MIN <= v <= STEP_SIZE_MAX):
            raise ValueError(f"step size {v} is outside plausible range (1–100).")
        return v


class PrepareMarkBase(BaseParameters):
    bounding_box_list: list[list[float]] | None = Field(
        None,
        description="Bounding box corners (4 × 2 array of [x, y] coordinates) "
        "defining a rectangular crop region used to determine the rotation of the image.",
    )
    mask_is_bitpacked: bool = Field(
        default=False,
        description="Whether the bytes in the mask are bit-packed. "
        'The expected bit-order for bit-packed arrays is "little".',
        examples=[True, False],
    )

    @field_validator("bounding_box_list")
    @classmethod
    def check_bounding_box_shape(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        """Check that the bounding box list has the correct shape."""
        if v is None:
            return v
        if len(v) != 4 or any(len(pt) != 2 for pt in v):  #noqa: PLR2004
            raise ValueError(
                f"bounding_box_list must be 4 [x, y] corner points, got {len(v)} points with "
                f"{[len(pt) for pt in v]} values in each point."
            )
        return v

    @cached_property
    def bounding_box(self) -> BoundingBox | None:
        """
        Convert the bounding_box tuple to a numpy array.

        :return: 2D numpy array of float values representing the bounding box
        """
        return np.array(self.bounding_box_list) if self.bounding_box_list is not None else None


class PrepareMarkStriation(PrepareMarkBase):
    mark_parameters: PreprocessingStriationParams = Field(..., description="Preprocessor parameters.")
    mark_type: MarkStriationType = Field(..., description="Type of mark to prepare.")


class PrepareMarkImpression(PrepareMarkBase):
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")
    mark_type: MarkImpressionType = Field(..., description="Type of mark to prepare.")


class EditImage(BaseParameters):
    """Request model for editing and transforming processed scan images."""

    cutoff_length: PositiveFloat = Field(
        description="Cutoff wavelength in meters (m) for Gaussian regression filtering. "
        "Defines the spatial frequency threshold for surface texture analysis.",
        examples=[1e-4, 2e-4, 2.5e-4],
    )
    resampling_factor: PositiveFloat = Field(
        default=4,
        description="Resampling rate for image resolution adjustment. Higher values increase resolution.",
        examples=[2, 4, 8],
    )
    surface_terms: SurfaceTermsAnnotated = Field(
        ...,
        description=(
            "Surface fitting model for leveling operations. PLANE for planar surfaces, SPHERE for curved surfaces. "
            "Accepts string (e.g. 'plane', 'PLANE') or int (e.g. 1) values."
        ),
        examples=["plane", "sphere"],
    )
    regression_order: RegressionOrder = Field(
        default=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        description="Polynomial regression order for surface fitting. R0 (constant), R1 (linear), or R2 (quadratic).",
    )
    crop: bool = Field(
        default=False,
        description="Whether to crop the image to the non-masked region.",
    )
    mask_is_bitpacked: bool = Field(
        default=False,
        description="Whether the bytes in the mask are bit-packed. "
        'The expected bit-order for bit-packed arrays is "little".',
        examples=[True, False],
    )

    @model_validator(mode="after")
    def check_file_is_x3p(self):
        """Check whether the scan file is an x3p file."""
        if self.scan_file.suffix.lower() != ".x3p":
            raise ValueError(f"Unsupported extension: {self.scan_file.suffix}")
        return self

    @field_validator("cutoff_length")
    @classmethod
    def check_plausible_cutoff(cls, v: float) -> float:
        """Check that the cutoff has a reasonable value."""
        if not (CUTOFF_MIN <= v <= CUTOFF_MAX):
            raise ValueError(
                f"cutoff_length {v} m is outside the plausible range ({CUTOFF_MIN}–{CUTOFF_MAX} m). "
                f"Value should be in meters, e.g. 250e-6, not 250."
            )
        return v

    @field_validator("resampling_factor")
    @classmethod
    def check_resampling_factor(cls, v: float) -> float:
        """Check that the resampling factor has a reasonable value."""
        if not (RESAMPLING_FACTOR_MIN <= v <= RESAMPLING_FACTOR_MAX):
            raise ValueError(
                f"resampling_factor {v} is outside plausible range ({RESAMPLING_FACTOR_MIN}-{RESAMPLING_FACTOR_MAX})."
            )
        return v


class GeneratedImages(URLContainer):
    preview_image: HttpUrl = Field(
        ...,
        description="RGBA preview image rendered from the parsed scan surface data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/preview.png"],
    )
    surface_map_image: HttpUrl = Field(
        ...,
        description="Height-map visualization of the scan surface.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/surface_map.png"],
    )


class PrepareMarkResponse(GeneratedImages):
    """Response model for prepared mark data access."""

    mark_data: HttpUrl = Field(
        ...,
        description="Cropped, rotated, and resampled mark data before surface filtering.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark.npz"],
    )
    mark_meta: HttpUrl = Field(
        ...,
        description="Metadata for the mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/mark.json"],
    )
    processed_data: HttpUrl = Field(
        ...,
        description="Mark surface data after filtering and preprocessing.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/processed.npz"],
    )
    processed_meta: HttpUrl = Field(
        ...,
        description="Metadata for the processed mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/processed.json"],
    )


class PrepareMarkResponseStriation(PrepareMarkResponse):
    """Response model for prepared striation mark data access."""

    profile_data: HttpUrl = Field(
        ...,
        description="Mean or median profile of a striation mark.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/profile.npz"],
    )


class PrepareMarkResponseImpression(PrepareMarkResponse):
    """Response model for prepared impression mark data access."""

    leveled_data: HttpUrl = Field(
        ...,
        description="Leveled impression mark surface (before surface filtering is applied).",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/leveled.npz"],
    )
    leveled_meta: HttpUrl = Field(
        ...,
        description="Metadata for the leveled impression mark data.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/leveled.json"],
    )


class ProcessedDataAccess(GeneratedImages):
    scan_image: HttpUrl = Field(
        ...,
        description="Subsampled X3P scan file, converted from the original upload.",
        examples=["http://localhost:8000/preprocessor/files/70fadc78-caf5-492a-a426-1cf2bf675f8c/scan.x3p"],
    )
