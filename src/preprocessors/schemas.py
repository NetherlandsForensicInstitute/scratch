from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from conversion.data_formats import BoundingBox, MarkImpressionType, MarkStriationType
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_striation import PreprocessingStriationParams
from pydantic import (
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from utils.constants import RegressionOrder

from models import (
    BaseModelConfig,
    ProjectTag,
    ScanFile,
    SupportedScanExtension,
)
from preprocessors.constants import SurfaceOptions
from schemas import URLContainer


def _update_schema(schema: dict[str, Any], attr_to_class: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    """Update the model JSON schema for correctly rendering the `openapi_extra` fields."""
    for attribute, class_name in attr_to_class:
        updated = schema["$defs"][class_name]
        for key in ("examples", "description"):
            if value := schema["properties"][attribute].get(key):
                updated[key] = value
        schema["properties"][attribute] = updated
    return schema


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

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        """Override the base method."""
        schema = super().model_json_schema(*args, **kwargs)
        attr_to_class = (
            ("scan_file", "ScanFile"),
            ("project_name", "ProjectTag"),
        )
        return _update_schema(schema, attr_to_class)


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
    mark_parameters: PreprocessingStriationParams = Field(..., description="Preprocessor parameters.")
    mark_type: MarkStriationType = Field(..., description="Type of mark to prepare.")

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        """Override the base method."""
        schema = super().model_json_schema(*args, **kwargs)
        attr_to_class = (("mark_parameters", "PreprocessingStriationParams"),)
        return _update_schema(schema, attr_to_class)


class PrepareMarkImpression(PrepareMarkBase):
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")
    mark_type: MarkImpressionType = Field(..., description="Type of mark to prepare.")

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        """Override the base method."""
        schema = super().model_json_schema(*args, **kwargs)
        attr_to_class = (("mark_parameters", "PreprocessingImpressionParams"),)
        return _update_schema(schema, attr_to_class)


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

    @model_validator(mode="after")
    def check_file_is_x3p(self):
        """Check whether the scan file is an x3p file."""
        if self.scan_file.suffix.lower() != ".x3p":
            raise ValueError(f"Unsupported extension: {self.scan_file.suffix}")
        return self

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        """Override the base method."""
        schema = super().model_json_schema(*args, **kwargs)
        # Add schema for BaseParameters and EditImage to JSON model
        attr_to_class = (
            ("regression_order", "RegressionOrder"),
            ("terms", "SurfaceOptions"),
        )
        return _update_schema(schema, attr_to_class)


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
