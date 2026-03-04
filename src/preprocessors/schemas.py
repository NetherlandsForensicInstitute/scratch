from __future__ import annotations

from functools import cached_property
from typing import Annotated, Any

import numpy as np
from conversion.data_formats import BoundingBox, MarkType
from conversion.preprocess_impression.parameters import PreprocessingImpressionParams
from conversion.preprocess_striation import PreprocessingStriationParams
from numpy.typing import NDArray
from pydantic import (
    AfterValidator,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from scipy.constants import micro
from utils.constants import RegressionOrder

from constants import MaskTypes
from models import (
    BaseModelConfig,
    ProjectTag,
    ScanFile,
    SupportedScanExtension,
)
from preprocessors.constants import SurfaceOptions


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
        1.0,
        gt=0.0,
        description="Horizontal pixel size in meters (m). Defines physical spacing between pixels in x-direction.",
        examples=[1.0, 0.5, 2.0],
    )
    scale_y: PositiveFloat = Field(
        1.0,
        description="Vertical pixel size in meters (m). Defines physical spacing between pixels in y-direction.",
        examples=[1.0, 0.5, 2.0],
    )
    step_size: PositiveInt = Field(
        1,
        description="Sets the sampling interval for both axes. "
        "Values > 1 downscale the image by skipping intermediate pixels.",
        examples=[1, 2, 4],
    )


class CropInfo(BaseModelConfig):
    type: MaskTypes
    data: dict
    is_foreground: bool


class PrepareMarkBase(BaseParameters):
    mark_type: MarkType = Field(..., description="Type of mark to prepare.")
    mask: list[list[float]] = Field(
        ...,
        description="2D boolean array representing the mask for the mark. Must have exactly the same shape"
        " (height × width) as the scan image.",
    )
    bounding_box_list: list[list[float]] | None = Field(
        None, description="Bounding box of a rectangular crop region used to determine the rotation of an image."
    )

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)

    @cached_property
    def bounding_box(self) -> BoundingBox | None:
        """
        Convert the bounding_box tuple to a numpy array.

        :return: 2D numpy array of float values representing the bounding box
        """
        return np.array(self.bounding_box_list) if self.bounding_box_list is not None else None


class PrepareMarkStriation(PrepareMarkBase):
    mark_parameters: PreprocessingStriationParams = Field(..., description="Preprocessor parameters.")

    @field_validator("mark_type")
    @classmethod
    def must_be_striation(cls, v: MarkType) -> MarkType:
        """Validate that the given mark type is a striation mark."""
        if not v.is_striation():
            raise ValueError(f"{v} is not a striation mark")
        return v


class PrepareMarkImpression(PrepareMarkBase):
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")

    @field_validator("mark_type")
    @classmethod
    def must_be_impression(cls, v: MarkType) -> MarkType:
        """Validate that the given mark type is an impression mark."""
        if not v.is_impression():
            raise ValueError(f"{v} is not an impression mark")
        return v


class MaskParameters(BaseModelConfig):
    shape: tuple[PositiveInt, PositiveInt] = Field(
        ...,
        examples=[[100, 100], [250, 150]],
        description="Shape (height, width) of the 2D mask array. Must exactly match the shape of the parsed scan"
        " image.",
    )
    is_bitpacked: bool = Field(default=False, examples=[False, True], description="Whether the mask is bit-packed.")


class EditImage(BaseParameters):
    """Request model for editing and transforming processed scan images."""

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
    terms: SurfaceOptions = Field(
        default=SurfaceOptions.PLANE,
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
    mask_parameters: MaskParameters = Field(
        ...,
        description="Mask parameters.",
        # TODO: change this field to `mask_shape: tuple[PositiveInt, PositiveInt]`
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
        # Add schema for mask parameters to JSON model
        schema["properties"]["mask_parameters"] = MaskParameters.model_json_schema(*args, **kwargs)
        # Add schema for BaseParameters and EditImage to JSON model
        attr_to_class = (
            ("scan_file", "ScanFile"),
            ("regression_order", "RegressionOrder"),
            ("terms", "SurfaceOptions"),
            ("project_name", "ProjectTag"),
        )
        for attribute, class_name in attr_to_class:
            updated = schema["$defs"][class_name]
            for key in ("examples", "description"):
                if value := schema["properties"][attribute].get(key):
                    updated[key] = value
            schema["properties"][attribute] = updated

        return schema
