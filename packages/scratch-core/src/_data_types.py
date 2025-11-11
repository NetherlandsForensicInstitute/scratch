from typing import Any, NamedTuple, Sequence
from numpy.typing import NDArray
from enum import StrEnum, auto
from pathlib import Path


class CropType(StrEnum):
    RECTANGLE = auto()
    LINE = auto()
    POLYGON = auto()
    CIRCLE = auto()


class DataType(StrEnum):
    SURFACE = auto()
    PROFILE = auto()
    IMAGE = auto()
    CIRCLE = auto()


class MarkType(NamedTuple):
    IMPRESSION = auto()
    STRIATION = auto()
    # TODO   % see list in Scratch 3.0 docu


class JavaStruct(NamedTuple):
    type: DataType
    mark_type: MarkType
    depth_data: NDArray[tuple[int, int]]  # % the z-data
    texture_data: NDArray[tuple[int, int]] | None  # % (color) RGB image
    quality_data: NDArray[tuple[int, int]] | None  # % greyscale image
    xdim: float  # % x pixel size
    ydim: float  # % y pixel size
    xdim_orig: float  # % x pixel size of the data after adding
    ydim_orig: float  # % y pixel size of the raw data when added to the database
    # Note that xdim_orig and ydim_orig may differ from the resolution the data was originally acquired, as the
    # data might have been subsampled when being added to the database, using the image_subsampling parameter
    invalid_pixel_val: float  # % value of invalid measurement point
    VR: float | None  # % vertical resolution
    LR: float | None  # % lateral resolution (for backward compatibility with Scratch 2)
    input_format: str  # % 'mat', 'x3p', 'al3d', 'png' ...
    additional_info: dict  # % all additional meta data
    crop_type: CropType  # % ROI type: e.g. 'rectangle', 'line', 'polygon' or 'circle'
    # (for backward compatibility with Scratch 2)
    crop_coordinates: list[
        tuple[int, int]
    ]  # % list of coordinates (n x 2) or structure
    # (for backward compatibility with Scratch 2)
    crop_info: Any  # % cell array containing inidividual steps of advanced cropping
    cutoff_hi: Any  # % the shape filter cutoff (in [um])
    cutoff_lo: Any  # % the noise filter cutoff (in [um])
    is_prep: bool  # % Indicates whether the data was pre-processed
    is_crop: bool  # % Indicates whether the data was cropped
    is_interp: bool  # % Indicates whether the data was interpolated
    is_resamp: bool  # % Indicates whether the data was resampled
    # (if so, the original sampling distance is still available in xdim_orig and ydim_orig)
    data_param: Any  # % Structure with pre-processing specific parameters
    subsampling: int | Sequence[int]  # % Image subsampling parameter
    orig_path: Path  # % The original data path
