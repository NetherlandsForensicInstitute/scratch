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

from models.enums import SupportedExtension
from models.image import ImageData

from .matfiles import load_mat_file
from .xthreep import load_x3p_file


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
    match extension := file_path.suffix.lower()[1:]:
        case SupportedExtension.MAT:
            return load_mat_file(file_path)
        case SupportedExtension.X3P:
            return load_x3p_file(file_path)
        # case SupportedExtension.AL3D:
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
            raise ValueError(f"Unsupported file format: {extension}")
