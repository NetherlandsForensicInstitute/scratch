from __future__ import annotations

from functools import cached_property
from typing import Annotated, Self

import numpy as np
from container_models.light_source import LightSource
from conversion.data_formats import BoundingBox
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
    pixel_size: float | None = Field(default=None, description="Physical target size of one pixel in meters.")
    adjust_pixel_spacing: bool = Field(
        default=True, description="Whether to adjust spacing between pixels during preprocessing."
    )
    level_offset: bool = Field(default=True, description="Apply offset leveling to remove constant height bias.")
    level_tilt: bool = Field(default=True, description="Apply tilt correction in X and Y directions.")
    level_2nd: bool = Field(
        default=True, description="Apply second-order leveling including astigmatism and defocus corrections."
    )
    interp_method: str = Field(
        default="cubic", description="Interpolation method for resampling ('nearest', 'linear', 'cubic', etc.)."
    )
    highpass_cutoff: float | None = Field(default=250.0e-6, description="High-pass filter cutoff frequency in meters.")
    lowpass_cutoff: float | None = Field(default=5.0e-6, description="Low-pass filter cutoff frequency in meters.")
    highpass_regression_order: int = Field(
        default=2, description="Polynomial order used for high-pass surface fitting."
    )
    lowpass_regression_order: int = Field(default=0, description="Polynomial order used for low-pass surface fitting.")

    @property
    def surface_terms(self) -> SurfaceTerms:
        """Convert leveling flags to SurfaceTerms."""
        terms = SurfaceTerms.NONE
        if self.level_offset:
            terms |= SurfaceTerms.OFFSET
        if self.level_tilt:
            terms |= SurfaceTerms.TILT_X | SurfaceTerms.TILT_Y
        if self.level_2nd:
            terms |= SurfaceTerms.ASTIG_45 | SurfaceTerms.DEFOCUS | SurfaceTerms.ASTIG_0
        return terms


class PreprocessingStriationParams(BaseModelConfig):
    highpass_cutoff: float = Field(
        default=2e-3, description="High-pass filter cutoff frequency for striation preprocessing in meters."
    )
    lowpass_cutoff: float = Field(
        default=2.5e-4, description="Low-pass filter cutoff frequency for striation preprocessing in meters."
    )
    cut_borders_after_smoothing: bool = Field(
        default=True, description="Whether to trim edges after smoothing to avoid border artifacts."
    )
    use_mean: bool = Field(default=True, description="Use mean value when calculating striation parameters.")
    angle_accuracy: float = Field(
        default=0.1, description="Accuracy threshold for determining striation angles in degrees."
    )
    max_iter: int = Field(default=25, description="Maximum number of iterations for angle fitting algorithm.")
    subsampling_factor: int = Field(default=1, description="Factor to reduce resolution for faster preprocessing.")


class PrepareMarkStriation(BaseParameters):
    mark_type: StriationMarks = Field(..., description="Type of mark to prepare.")
    mask: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    bounding_box_list: list[list[float]] | None = Field(
        None, description="Bounding box of a rectangular crop region used to determine the rotation of an image."
    )
    mark_parameters: PreprocessingStriationParams = Field(..., description="Preprocessor parameters.")

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """
        Convert the bounding_box tuple to a numpy array.

        :return: 2D numpy array of boolean values representing the bounding box
        """
        return np.array(self.bounding_box_list)


class PrepareMarkImpression(BaseParameters):
    mark_type: ImpressionMarks = Field(..., description="Type of mark to prepare.")
    mask: list[list[float]] = Field(..., description="Array representing the mask for the mark.")
    bounding_box_list: list[list[float]] | None = Field(
        None, description="Bounding box of a rectangular crop region used to determine the rotation of an image."
    )
    mark_parameters: PreprocessingImpressionParams = Field(..., description="Preprocessor parameters.")

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """
        Convert the bounding_box tuple to a numpy array.

        :return: 2D numpy array of boolean values representing the bounding box
        """
        return np.array(self.bounding_box_list)


type Mask = tuple[tuple[bool, ...], ...]


class EditImage(BaseParameters):
    """Request model for editing and transforming processed scan images."""

    mask: Mask = Field(
        description=(
            "Binary mask defining regions to include (True/1) or exclude (False/0) during processing. "
            "Accepts both boolean (True/False) and integer (1/0) representations. "
            "Must be a 2D tuple structure matching the scan dimensions."
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
        if self.scan_file.suffix != ".x3p":
            raise ValueError(f"Unsupported extension: {self.scan_file.suffix}")
        return self

    @cached_property
    def mask_array(self) -> NDArray:
        """
        Convert the mask tuple to a numpy boolean array.

        :return: 2D numpy array of boolean values representing the mask
        """
        return np.array(self.mask, np.bool_)
