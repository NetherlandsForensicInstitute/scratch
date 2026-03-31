from __future__ import annotations

from functools import cached_property

import numpy as np
from conversion.data_formats import BoundingBox, MarkType
from fastapi import File, UploadFile
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
    ScanFile,
    SupportedScanExtension,
)
from preprocessors.constants import SurfaceOptions
from schemas import URLContainer


class BaseParameters(BaseModelConfig):
    """Base parameters for preprocessor operations including scan file."""

    project_name: str | None = Field(
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


class PrepareMarkBase(BaseParameters):
    mark_type: MarkType
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

    @cached_property
    def bounding_box(self) -> BoundingBox | None:
        """
        Convert the bounding_box tuple to a numpy array.

        :return: 2D numpy array of float values representing the bounding box
        """
        return np.array(self.bounding_box_list) if self.bounding_box_list is not None else None


class PrepareMarkStriation(PrepareMarkBase):
    highpass_cutoff: float = 2e-3
    lowpass_cutoff: float = 2.5e-4
    cut_borders_after_smoothing: bool = True
    use_mean: bool = True
    angle_accuracy: float = 0.1
    max_iter: int = 25
    subsampling_factor: int = 1
    mask_data: UploadFile = File(
        ..., description="Mask given as binary data. The shape of the mask needs to be the same as scan_image."
    )

    @field_validator("mark_type")
    @classmethod
    def must_be_striation(cls, v: MarkType) -> MarkType:
        """Validate that the given mark type is a striation mark."""
        if not v.is_striation():
            raise ValueError(f"{v} is not a striation mark")
        return v


class PrepareMarkImpression(PrepareMarkBase):
    pixel_size: float | None = None
    adjust_pixel_spacing: bool = True
    level_offset: bool = True
    level_tilt: bool = True
    level_2nd: bool = True
    interp_method: str = "cubic"
    highpass_cutoff: float | None = 250.0e-6
    lowpass_cutoff: float | None = 5.0e-6
    highpass_regression_order: int = 2
    lowpass_regression_order: int = 0
    mask_data: UploadFile = File(
        ..., description="Mask given as binary data. The shape of the mask needs to be the same as scan_image."
    )

    @field_validator("mark_type")
    @classmethod
    def must_be_impression(cls, v: MarkType) -> MarkType:
        """Validate that the given mark type is an impression mark."""
        if not v.is_impression():
            raise ValueError(f"{v} is not an impression mark")
        return v


class EditImage(BaseParameters):
    """Request model for editing and transforming processed scan images."""

    cutoff_length: PositiveFloat = Field(
        description="Cutoff wavelength in micrometers (µm) for Gaussian regression filtering. "
        "Defines the spatial frequency threshold for surface texture analysis.",
        examples=[250, 500, 1000],
    )
    resampling_factor: PositiveFloat = Field(
        default=4,
        description="Resampling rate for image resolution adjustment. Higher values increase resolution.",
        examples=[2, 4, 8],
    )
    terms: SurfaceOptions = Field(
        ...,
        description=(
            "Surface fitting model for leveling operations. PLANE for planar surfaces, SPHERE for curved surfaces."
        ),
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
    mask_data: UploadFile = File(
        ..., description="Mask given as binary data. The shape of the mask needs to be the same as scan_image."
    )

    @model_validator(mode="after")
    def check_file_is_x3p(self):
        """Check whether the scan file is an x3p file."""
        if self.scan_file.suffix.lower() != ".x3p":
            raise ValueError(f"Unsupported extension: {self.scan_file.suffix}")
        return self


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
