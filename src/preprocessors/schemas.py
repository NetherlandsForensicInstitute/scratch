from __future__ import annotations

from functools import cached_property
from typing import Annotated, Any

import numpy as np
from container_models.light_source import LightSource
from conversion.leveling.data_types import SurfaceTerms
from numpy.typing import NDArray
from pydantic import (
    AfterValidator,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from scipy.constants import micro
from utils.constants import RegressionOrder

from constants import LIGHT_SOURCES, OBSERVER, ImpressionMarks, MaskTypes, StriationMarks
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
        LIGHT_SOURCES,
        description="Light sources for surface illumination rendering.",
        examples=[
            (
                LightSource(azimuth=90, elevation=45),
                LightSource(azimuth=180, elevation=45),
            ),
        ],
    )
    observer: LightSource = Field(
        OBSERVER,
        description="Observer viewpoint vector for surface rendering.",
        examples=[LightSource(azimuth=90, elevation=45)],
    )
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
    step_size_x: PositiveInt = Field(
        1,
        description="Subsampling step in x-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )
    step_size_y: PositiveInt = Field(
        1,
        description="Subsampling step in y-direction. Values > 1 reduce resolution by skipping pixels.",
        examples=[1, 2, 4],
    )


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
    mask: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    rotation_angle: int = Field(0, description="Rotation angle for the mark preparation.")
    crop_info: CropInfo | None = Field(
        None, description="", examples=[{"type": "rectangle", "data": {}, "is_foreground": False}]
    )
    mark_parameters: PreprocessingStriationParams = Field(
        ..., description="Preprocessor parameters."
    )  # TODO: not yet merged dataclass from PR #84

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)


class PrepareMarkImpression(BaseParameters):
    mark_type: ImpressionMarks = Field(..., description="Type of mark to prepare.")
    mask: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    rotation_angle: int = Field(0, description="Rotation angle for the mark preparation.")
    crop_info: CropInfo | None = Field(
        None, description="", examples=[{"type": "rectangle", "data": {}, "is_foreground": False}]
    )
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)


class MaskParameters(BaseModelConfig):
    shape: tuple[PositiveInt, PositiveInt] = Field(
        ..., examples=[[100, 100], [250, 150]], description="Defines the shape of the 2D mask array."
    )
    is_bitpacked: bool = Field(
        default=False, examples=[False, True], description="Whether the mask is bit-packed."
    )  # TODO: create enum/flags for compression types


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
    terms: SurfaceTerms = Field(
        default=SurfaceTerms.PLANE,
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
    mask_parameters: MaskParameters = Field(..., description="Mask parameters.")

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
            ("terms", "SurfaceTerms"),
            ("project_name", "ProjectTag"),
        )
        for attribute, class_name in attr_to_class:
            updated = schema["$defs"][class_name]
            for key in ("examples", "description"):
                if value := schema["properties"][attribute].get(key):
                    updated[key] = value
            schema["properties"][attribute] = updated

        return schema
