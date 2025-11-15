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

from scipy.io import loadmat

from models.enums import ImageType, InputFormat
from models.image import ImageData


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

    Raises
    ------
    ValueError
        If required fields are missing or invalid
    """
    # Load the MAT file
    mat_data = loadmat(file_path)

    # Check if 'type' field exists
    if "type" not in mat_data:
        raise ValueError("MAT file must contain 'type' field")

    # Get the type and map to ImageType enum
    # scipy.io.loadmat returns strings as arrays, so we need to extract the value
    type_val = mat_data["type"]
    if isinstance(type_val, np.ndarray):
        # Handle both scalar and array cases
        type_str = str(type_val.flat[0])
    else:
        type_str = str(type_val)

    image_type = ImageType(type_str.lower())

    # Extract depth_data (required for most types)
    depth_data = mat_data.get("depth_data")

    # scipy.io.loadmat converts 1D arrays to 2D row vectors (1, n)
    # For profiles, we need to flatten them back to 1D
    if depth_data is not None and depth_data.ndim == 2 and depth_data.shape[0] == 1:
        depth_data = depth_data.flatten()

    # Extract dimensions
    xdim = mat_data.get("xdim", 0.0)
    if isinstance(xdim, np.ndarray):
        xdim = float(xdim.item())
    else:
        xdim = float(xdim)

    ydim = mat_data.get("ydim", 0.0)
    if isinstance(ydim, np.ndarray):
        ydim = float(ydim.item())
    else:
        ydim = float(ydim)

    # Extract optional texture_data
    texture_data = mat_data.get("texture_data")

    return ImageData(
        type=image_type,
        depth_data=depth_data,
        texture_data=texture_data,
        xdim=xdim,
        ydim=ydim,
    )


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
        # case InputFormat.X3P:
        #     return _load_x3p_file(file_path)
        # case InputFormat.X3P:
        #     return _load_al3d_file(file_path)
        # case InputFormat.SUR:
        #     return _load_sur_file(file_path)
        # case InputFormat.LMS:
        #     return _load_lms_file(file_path)
        # case InputFormat.PLU:
        #     return _load_plu_file(file_path)
        # case InputFormat.PNG | InputFormat.BMP | InputFormat.JPG | InputFormat.JPEG:
        #     return _load_image_file(file_path)
        case _:
            raise ValueError(f"Unsupported file format: {ext}")
