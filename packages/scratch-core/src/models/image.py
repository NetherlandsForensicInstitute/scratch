from typing import Any

from numpy import nan
from pydantic import Field

from .base import (
    FloatArray1D,
    FloatArray2D,
    FrozenBaseModel,
    ImageArray2D,
    ImageArray3D,
)
from .enums import CropType, ImageType, SupportedExtension


class ImageData(FrozenBaseModel):
    """
    Output structure matching MATLAB data_out.
    """

    type: ImageType
    mark_type: str = Field(default="", description="See list in Scratch 3.0 docu")
    depth_data: FloatArray2D | FloatArray1D | None = Field(
        default=None, description="The z-data (2D for surfaces, 1D for profiles)"
    )
    texture_data: ImageArray3D | ImageArray2D | None = Field(
        default=None, description="(color) RGB image (H×W×3) or grayscale (H×W)"
    )
    quality_data: ImageArray2D | None = Field(
        default=None, description="Greyscale image (H×W)"
    )
    xdim: float = Field(default=0.0, description="X pixel size in meters")
    ydim: float = Field(default=0.0, description="Y pixel size in meters")
    xdim_orig: float = Field(
        default=0.0,
        description="X pixel size of the data after adding it to the database",
    )
    ydim_orig: float = Field(
        default=0.0,
        description="Y pixel size of the raw data when added to the database",
    )
    invalid_pixel_val: float = Field(
        default=nan, description="Value of invalid measurement point"
    )
    vertical_resolution: float | None = Field(
        default=None, description="Vertical resolution"
    )
    lateral_resolution: float | None = Field(
        default=None, description="Lateral resolution"
    )
    input_format: SupportedExtension | None = None
    additional_info: dict[str, Any] = Field(
        default_factory=dict, description="All additional meta data"
    )
    crop_type: CropType | None = None
    crop_coordinates: list | dict | None = Field(
        default=None, description="List of coordinates (n x 2) or structure"
    )
    crop_info: list = Field(
        default_factory=list,
        description="List containing individual steps of advanced cropping",
    )
    cutoff_hi: list = Field(
        default_factory=list, description="The shape filter cutoff (in [um])"
    )
    cutoff_lo: list = Field(
        default_factory=list, description="The noise filter cutoff (in [um])"
    )
    is_prep: bool = Field(
        default=False, description="Indicates whether the data was pre-processed"
    )
    is_crop: bool = Field(
        default=False, description="Indicates whether the data was cropped"
    )
    is_interp: bool = Field(
        default=False, description="Indicates whether the data was interpolated"
    )
    is_resamp: bool = Field(
        default=False, description="Indicates whether the data was resampled"
    )
    data_param: dict[str, Any] = Field(
        default_factory=dict,
        description="Structure with pre-processing specific parameters",
    )
    subsampling: int = Field(default=1, description="Image subsampling parameter")
    orig_path: str = Field(default="", description="The original data path")
