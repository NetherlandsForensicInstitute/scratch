"""
Translation of importdata.m to Python.

This is a generic data import routine for all formats that are relevant:
{'*.mat', '*.al3d', '*.x3p', '*.sur', '*.lms', '*.plu', '*.png', '*.bmp', '*.jpg', '*.jpeg'}

NOTE: For most formats, not all functionality is implemented! For example
for *.plu datasets, only surfaces (MES_TOPO) can be read. For 2D images,
there MUST be either an image of a ruler (ruler.*) with stripes of 1 mm
distance in the same directory as the 2D image or a text file
sampling_distance.txt with the sampling distance in micrometers (e.g. 4.26)
at the beginning, followed by a space!
"""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field

from .data_types import FrozenBaseModel
from scipy.io import loadmat

from surfalize import Surface


class DataType(StrEnum):
    SURFACE = auto()
    PROFILE = auto()
    IMAGE = auto()


class InputFormat(StrEnum):
    MAT = auto()
    AL3D = auto()
    X3P = auto()
    SUR = auto()
    LMS = auto()
    PLU = auto()
    PNG = auto()
    BMP = auto()
    JPG = auto()
    JPEG = auto()


class CropType(StrEnum):
    RECTANGLE = auto()
    LINE = auto()
    POLYGON = auto()
    CIRCLE = auto()


class DataOutput(FrozenBaseModel):
    """
    Output structure matching MATLAB data_out.
    """

    type: DataType
    mark_type: str = Field(default="", description="See list in Scratch 3.0 docu")
    depth_data: NDArray | None = Field(default=None, description="The z-data")
    texture_data: NDArray | None = Field(default=None, description="(color) RGB image")
    quality_data: NDArray | None = Field(default=None, description="Greyscale image")
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
        default=np.nan, description="Value of invalid measurement point"
    )
    vertical_resolution: float | None = Field(
        default=None, description="Vertical resolution"
    )
    lateral_resolution: float | None = Field(
        default=None, description="Lateral resolution"
    )
    input_format: InputFormat
    additional_info: dict[str, Any] = Field(
        default_factory=dict, description="All additional meta data"
    )
    crop_type: CropType
    crop_coordinates: list | dict | None = Field(
        default=None, description="List of coordinates (n x 2) or structure"
    )
    crop_info: list = Field(
        default_factory=list,
        description="List containing individual steps of advanced cropping",
    )
    cutoff_hi: float | None = Field(
        default=None, description="The shape filter cutoff (in [um])"
    )
    cutoff_lo: float | None = Field(
        default=None, description="The noise filter cutoff (in [um])"
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


# TODO MAT files for testing
def _load_mat_file(file_path: Path) -> DataOutput:
    """
    Load MAT file format.

    Parameters
    ----------
    file_path : Path
        Path to the .mat file

    Returns
    -------
    DataOutput
        Loaded data structure
    """

    data = loadmat(str(file_path))
    # Get the first field (similar to MATLAB tmp = fields(data))

    if not (field_names := [k for k in data.keys() if not k.startswith("__")]):
        raise ValueError(f"No data fields found in {file_path}")

    data_struct = data[field_names[0]]

    # Convert MATLAB struct to Python dict if needed
    if hasattr(data_struct, "dtype") and hasattr(data_struct.dtype, "names"):
        # It's a structured array
        data_dict = {name: data_struct[name][0, 0] for name in data_struct.dtype.names}
    else:
        data_dict = data_struct

    # Create output structure
    data_out = DataOutput(**data_dict)

    # Handle backward compatibility with Scratch 2.0
    # selection_type and crop_type logic
    selection_type = data_dict.get("selection_type", "")
    select_coordinates = data_dict.get("select_coordinates", None)
    crop_type = data_dict.get("crop_type", "")
    crop_coordinates = data_dict.get("crop_coordinates", None)

    if selection_type and crop_type:
        if selection_type == crop_type:
            if (select_coordinates is None and crop_coordinates is not None) or (
                select_coordinates is not None and crop_coordinates is None
            ):
                data_out.crop_coordinates = select_coordinates
            elif select_coordinates is not None and crop_coordinates is not None:
                tmp1 = np.asarray(crop_coordinates).flatten()
                tmp2 = np.asarray(select_coordinates).flatten()
                if len(tmp1) == len(tmp2):
                    if not np.array_equal(tmp1, tmp2):
                        data_out.crop_coordinates = select_coordinates
                else:
                    data_out.crop_coordinates = select_coordinates
        else:
            data_out.crop_type = str(selection_type)
            data_out.crop_coordinates = select_coordinates
    else:
        if selection_type:
            data_out.crop_type = str(selection_type)
            data_out.crop_coordinates = select_coordinates
        else:
            data_out.crop_type = crop_type if crop_type else ""
            data_out.crop_coordinates = crop_coordinates

    if "crop_info" not in data_dict:
        data_out.crop_info = []
    else:
        data_out.crop_info = list(data_dict["crop_info"])

    if "cutoff_hi" in data_dict:
        data_out.cutoff_hi = data_dict["cutoff_hi"]

    if "cutoff_lo" in data_dict:
        data_out.cutoff_lo = data_dict["cutoff_lo"]

    if "is_prep" in data_dict:
        data_out.is_prep = int(data_dict["is_prep"])

    if "is_crop" in data_dict:
        data_out.is_crop = int(data_dict["is_crop"])

    if "is_interp" in data_dict:
        data_out.is_interp = int(data_dict["is_interp"])

    if "is_resamp" in data_dict:
        data_out.is_resamp = int(data_dict["is_resamp"])

    if "data_param" in data_dict:
        data_out.data_param = dict(data_dict["data_param"])

    if "subsampling" in data_dict:
        data_out.subsampling = int(data_dict["subsampling"])

    if "orig_path" not in data_dict:
        data_out.orig_path = str(file_path)
    else:
        data_out.orig_path = str(data_dict["orig_path"])

    data_out.input_format = InputFormat.MAT
    return data_out


def _load_x3p_file(file_path: Path) -> DataOutput:
    """
    Load X3P file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .x3p file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    surface = Surface.load(file_path)

    data_out = DataOutput(
        depth_data=surface.data.T if surface.data_shape[1] > 1 else surface.data,
        texture_data=None,
        quality_data=None,
        xdim=surface.step_x * 1e-6,
    )

    # Determine type based on feature type
    if surface.data.shape[1] > 1:
        data_out.type = DataType.SURFACE
        # surfalize returns step_x and step_y in micrometers, convert to meters
        data_out.ydim = surface.step_y * 1e-6
    else:
        data_out.type = DataType.PROFILE
        data_out.ydim = None

    data_out.vertical_resolution = None
    data_out.lateral_resolution = None
    data_out.additional_info = {"metadata": surface.metadata}
    data_out.input_format = InputFormat.X3P
    data_out.invalid_pixel_val = np.nan
    data_out.orig_path = str(file_path)

    return data_out


def _load_al3d_file(file_path: Path) -> DataOutput:
    """
    Load AL3D (Alicona) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .al3d file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    surface = Surface.load(file_path)

    data_out = DataOutput()
    data_out.type = DataType.SURFACE

    # surfalize already handles conversion to micrometers
    # and replaces invalid values with NaN
    data_out.depth_data = np.asarray(surface.data, dtype=np.float64)

    # surfalize doesn't return texture/quality for AL3D
    data_out.texture_data = None
    data_out.quality_data = None

    # Convert from micrometers to meters
    data_out.xdim = surface.step_x * 1e-6
    data_out.ydim = surface.step_y * 1e-6

    # Try to extract LR and VR from metadata if available
    data_out.vertical_resolution = None
    data_out.lateral_resolution = None

    if hasattr(surface, "metadata") and surface.metadata:
        data_out.additional_info = {"metadata": surface.metadata}
    else:
        data_out.additional_info = {}

    data_out.input_format = InputFormat.AL3D
    data_out.invalid_pixel_val = np.nan
    data_out.orig_path = str(file_path)

    return data_out


def _load_sur_file(file_path: Path) -> DataOutput:
    """
    Load SUR (Mountains Map) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .sur file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return DataOutput(
        type=DataType.SURFACE,
        depth_data=np.asarray(surface.data, dtype=np.float64),
        texture_data=None,
        quality_data=None,
        # Convert from micrometers to meters
        xdim=surface.step_x * 1e-6,
        ydim=surface.step_y * 1e-6,
        vertical_resolution=None,
        lateral_resolution=None,
        additional_info={"metadata": surface.metadata}
        if hasattr(surface, "metadata") and surface.metadata
        else {},
        input_format=InputFormat.SUR,
        invalid_pixel_val=np.nan,
        orig_path=str(file_path),
    )


def _load_lms_file(file_path: Path) -> DataOutput:
    """
    Load LMS (Zeiss) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .lms file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return DataOutput(
        type=DataType.SURFACE,
        depth_data=np.asarray(surface.data, dtype=np.float64),
        # As in MATLAB, texture is removed by default for LMS
        texture_data=None,
        quality_data=None,
        # Convert from micrometers to meters
        xdim=surface.step_x * 1e-6,
        ydim=surface.step_y * 1e-6,
        vertical_resolution=None,
        lateral_resolution=None,
        additional_info={"metadata": surface.metadata}
        if hasattr(surface, "metadata") and surface.metadata
        else {},
        input_format=InputFormat.LMS,
        invalid_pixel_val=np.nan,
        orig_path=str(file_path),
    )


def _load_plu_file(file_path: Path) -> DataOutput:
    """
    Load PLU (Sensofar) file format using surfalize.

    NOTE: Only the MES_TOPO data type is implemented!

    Parameters
    ----------
    file_path : Path
        Path to the .plu file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return DataOutput(
        type=DataType.SURFACE,
        depth_data=np.asarray(surface.data, dtype=np.float64),
        texture_data=None,
        quality_data=None,
        # Convert from micrometers to meters
        xdim=surface.step_x * 1e-6,
        ydim=surface.step_y * 1e-6,
        vertical_resolution=None,
        lateral_resolution=None,
        additional_info={"metadata": surface.metadata}
        if hasattr(surface, "metadata") and surface.metadata
        else {},
        input_format=InputFormat.PLU,
        invalid_pixel_val=np.nan,
        orig_path=str(file_path),
    )


def _determine_2d_image_sampling_distance(ruler_image: NDArray) -> float:
    """
    Determine sampling distance from a ruler image.

    This is a placeholder for the actual implementation.
    The MATLAB version has a function Determine2DImageSamplingDistance
    that analyzes ruler stripes.

    Parameters
    ----------
    ruler_image : NDArray
        Image of a ruler with stripes of 1 mm distance

    Returns
    -------
    float
        Sampling distance in micrometers
    """
    msg = "Ruler-based sampling distance determination not yet implemented"
    raise NotImplementedError(msg)


def _rgb_to_grey(rgb_image: NDArray) -> NDArray:
    """
    Convert RGB image to greyscale.

    Parameters
    ----------
    rgb_image : NDArray
        RGB image array

    Returns
    -------
    NDArray
        Greyscale image
    """
    if len(rgb_image.shape) == 3:
        # Standard RGB to greyscale conversion
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    return rgb_image


def _stretch(
    image: NDArray, plow: float, phigh: float, low: float, high: float
) -> NDArray:
    """
    Stretch image contrast.

    Parameters
    ----------
    image : NDArray
        Input image
    plow : float
        Lower percentile
    phigh : float
        Upper percentile
    low : float
        Output lower bound
    high : float
        Output upper bound

    Returns
    -------
    NDArray
        Stretched image
    """
    plow_val = np.percentile(image, plow)
    phigh_val = np.percentile(image, phigh)

    stretched = np.clip(image, plow_val, phigh_val)
    stretched = (stretched - plow_val) / (phigh_val - plow_val)
    return stretched * (high - low) + low


def _load_image_file(file_path: Path) -> DataOutput:
    """
    Load 2D image file (PNG, BMP, JPG, JPEG).

    Parameters
    ----------
    file_path : Path
        Path to the image file

    Returns
    -------
    DataOutput
        Loaded data structure
    """
    data_out = DataOutput()
    data_out.type = "image"

    # Read image
    img = Image.open(file_path)
    data_array = np.asarray(img)

    # Convert to greyscale if RGB
    if len(data_array.shape) == 3:
        depth_data = _rgb_to_grey(data_array)
        texture_data = data_array
    else:
        depth_data = data_array.astype(np.float64)
        texture_data = data_array

    # Flip vertically (MATLAB indexing difference)
    depth_data = np.flipud(depth_data)
    texture_data = np.flipud(texture_data)

    # Determine resolution
    resolution_set = False
    xdim = 0.0

    parent_dir = file_path.parent

    # Check for ruler image
    if not resolution_set:
        ruler_files = ["ruler.png", "ruler.jpg", "ruler.bmp"]
        for ruler_file in ruler_files:
            ruler_path = parent_dir / ruler_file
            if ruler_path.exists():
                ruler_img = np.asarray(Image.open(ruler_path))
                try:
                    xdim = _determine_2d_image_sampling_distance(ruler_img)
                    resolution_set = True
                    break
                except NotImplementedError:
                    pass

    # Check for sampling_distance.txt
    if not resolution_set:
        sampling_file = parent_dir / "sampling_distance.txt"
        if sampling_file.exists():
            with sampling_file.open() as f:
                content = f.read()

            # Find the line terminator '%', remove spaces and change ',' to '.'
            if "%" in content:
                tmp = content.split("%")[0]
            else:
                # Look for LF and CR
                lines = content.split("\n")
                if lines:
                    tmp = lines[0].split("\r")[0]
                else:
                    tmp = content

            # Remove spaces and replace comma with period
            tmp = tmp.replace(" ", "").replace(",", ".")
            try:
                xdim = float(tmp)
                resolution_set = True
            except ValueError:
                pass

    if xdim == 0:
        msg = "Sampling distance not defined for 2D image!"
        raise ValueError(msg)

    # Normalize the intensity data and scale to 'toolmark level'
    depth_data_stretch = _stretch(depth_data, 1, 99, 0, 255)
    depth_data_stretch_scale = depth_data_stretch - np.median(depth_data_stretch)
    depth_data = depth_data_stretch_scale

    # Convert from micrometers to meters
    xdim_m = xdim * 1e-6
    ydim_m = xdim_m

    data_out.depth_data = depth_data
    data_out.texture_data = texture_data
    data_out.quality_data = None
    data_out.xdim = xdim_m
    data_out.ydim = ydim_m
    data_out.vertical_resolution = None
    data_out.lateral_resolution = None
    data_out.additional_info = {}
    data_out.input_format = file_path.suffix[1:].lower()
    data_out.invalid_pixel_val = np.nan
    data_out.orig_path = str(file_path)

    return data_out


def import_data(file_path: Path) -> DataOutput:
    """
    Generic data import routine for all supported formats.

    Supported formats:
    - MAT files (*.mat)
    - X3P files (*.x3p)
    - AL3D files (*.al3d) - Alicona surfaces
    - SUR files (*.sur) - Mountains Map surfaces
    - LMS files (*.lms) - Zeiss surfaces
    - PLU files (*.plu) - Sensofar surfaces
    - Image files (*.png, *.bmp, *.jpg, *.jpeg)

    Parameters
    ----------
    file_path : Path
        Path to the file to import.

    Returns
    -------
    DataOutput
        Data structure containing the loaded data and metadata

    Raises
    ------
    ValueError
        If file format is not supported or file does not exist

    Examples
    --------
    >>> data = import_data("path/to/surface.x3p")
    >>> print(data.type)
    'surface'
    >>> print(data.depth_data.shape)
    (512, 512)
    """

    # Dispatch to appropriate loader based on extension
    match ext := file_path.suffix.lower()[1:]:
        case InputFormat.MAT:
            return _load_mat_file(file_path)
        case InputFormat.X3P:
            return _load_x3p_file(file_path)
        case InputFormat.X3P:
            return _load_al3d_file(file_path)
        case InputFormat.SUR:
            return _load_sur_file(file_path)
        case InputFormat.LMS:
            return _load_lms_file(file_path)
        case InputFormat.PLU:
            return _load_plu_file(file_path)
        case InputFormat.PNG | InputFormat.BMP | InputFormat.JPG | InputFormat.JPEG:
            return _load_image_file(file_path)
        case _:
            raise ValueError(f"Unsupported file format: {ext}")
