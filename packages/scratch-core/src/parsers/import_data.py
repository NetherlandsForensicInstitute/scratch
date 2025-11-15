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

from pathlib import Path

import numpy as np
from PIL import Image

from surfalize import Surface
from scipy.io import loadmat

from models.enums import ImageType, InputFormat
from models.image import ImageData, ImageArray2D, FloatArray2D


# TODO MAT files for testing
def _load_mat_file(file_path: Path) -> ImageData:
    """
    Load MAT file format.

    Parameters
    ----------
    file_path : Path
        Path to the .mat file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    loadmat(file_path)
    return ImageData()


def _load_x3p_file(file_path: Path) -> ImageData:
    """
    Load X3P file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .x3p file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    surface = Surface.load(file_path)

    data_out = ImageData(
        depth_data=surface.data.T if surface.data_shape[1] > 1 else surface.data,
        texture_data=None,
        quality_data=None,
        xdim=surface.step_x * 1e-6,
        vertical_resolution=None,
        lateral_resolution=None,
        additional_info={"metadata": surface.metadata},
        input_format=InputFormat.X3P,
        invalid_pixel_val=np.nan,
        orig_path=str(file_path),
    )

    # Determine type based on feature type
    if surface.data.shape[1] > 1:
        data_out.type = ImageType.SURFACE
        # surfalize returns step_x and step_y in micrometers, convert to meters
        data_out.ydim = surface.step_y * 1e-6
    else:
        data_out.type = ImageType.PROFILE
        data_out.ydim = None

    return data_out


def _load_al3d_file(file_path: Path) -> ImageData:
    """
    Load AL3D (Alicona) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .al3d file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return ImageData(
        type=ImageType.SURFACE,
        depth_data=np.asarray(surface.data, dtype=np.float64),
        texture_data=None,
        quality_data=None,
        xdim=surface.step_x * 1e-6,
        ydim=surface.step_y * 1e-6,
        vertical_resolution=None,
        lateral_resolution=None,
        additional_info={"metadata": surface.metadata}
        if hasattr(surface, "metadata") and surface.metadata
        else {},
        input_format=InputFormat.AL3D,
        invalid_pixel_val=np.nan,
        orig_path=str(file_path),
    )


def _load_sur_file(file_path: Path) -> ImageData:
    """
    Load SUR (Mountains Map) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .sur file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return ImageData(
        type=ImageType.SURFACE,
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


def _load_lms_file(file_path: Path) -> ImageData:
    """
    Load LMS (Zeiss) file format using surfalize.

    Parameters
    ----------
    file_path : Path
        Path to the .lms file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return ImageData(
        type=ImageType.SURFACE,
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


def _load_plu_file(file_path: Path) -> ImageData:
    """
    Load PLU (Sensofar) file format using surfalize.

    NOTE: Only the MES_TOPO data type is implemented!

    Parameters
    ----------
    file_path : Path
        Path to the .plu file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    surface = Surface.load(file_path)

    return ImageData(
        type=ImageType.SURFACE,
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


def _determine_2d_image_sampling_distance(
    ruler_image: ImageArray2D,
) -> float:
    """
    Determine sampling distance from a ruler image.

    This is a placeholder for the actual implementation.
    The MATLAB version has a function Determine2DImageSamplingDistance
    that analyzes ruler stripes.

    Parameters
    ----------
    ruler_image : ImageArray2D | ImageArray3D
        Image of a ruler with stripes of 1 mm distance

    Returns
    -------
    float
        Sampling distance in micrometers
    """
    msg = "Ruler-based sampling distance determination not yet implemented"
    raise NotImplementedError(msg)


def _rgb_to_grey(rgb_image: ImageArray2D) -> FloatArray2D:
    """
    Convert RGB image to greyscale.

    Parameters
    ----------
    rgb_image : ImageArray2D | ImageArray3D
        RGB image array (H×W×3) or grayscale (H×W)

    Returns
    -------
    FloatArray2D
        Greyscale image (H×W)
    """
    if len(rgb_image.shape) == 3:
        # Standard RGB to greyscale conversion
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    return rgb_image.astype(np.float64)


def _stretch(
    image: FloatArray2D, plow: float, phigh: float, low: float, high: float
) -> FloatArray2D:
    """
    Stretch image contrast.

    Parameters
    ----------
    image : FloatArray2D
        Input image (H×W)
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
    FloatArray2D
        Stretched image (H×W)
    """
    plow_val = np.percentile(image, plow)
    phigh_val = np.percentile(image, phigh)

    stretched = np.clip(image, plow_val, phigh_val)
    stretched = (stretched - plow_val) / (phigh_val - plow_val)
    return stretched * (high - low) + low


def _load_image_file(file_path: Path) -> ImageData:
    """
    Load 2D image file (PNG, BMP, JPG, JPEG).

    Parameters
    ----------
    file_path : Path
        Path to the image file

    Returns
    -------
    ImageData
        Loaded data structure
    """
    data_out = ImageData()
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


def import_data(file_path: Path) -> ImageData:
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
    ImageData
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
