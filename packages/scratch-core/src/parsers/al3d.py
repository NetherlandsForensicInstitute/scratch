"""AL3D file format parser.

AL3D is Alicona's proprietary format for storing surface topography data.
This parser translates the MATLAB AliconaReader functionality to Python.
"""

from pathlib import Path

import numpy as np
from scipy.constants import micro
from surfalize import Surface

from models.enums import ImageType, SupportedExtension
from models.image import ImageData

from .extract_al3d_resolution import extract_resolutions_from_xml_data


def load_al3d_file(file_path: Path) -> ImageData:
    """
    Load Alicona AL3D file format.

    Parameters
    ----------
    file_path : Path
        Path to the .al3d file

    Returns
    -------
    ImageData
        Loaded data structure matching MATLAB data_out format

    Raises
    ------
    ValueError
        If required fields are missing or invalid
    """
    # Load the surface using surfalize with our patched reader
    # The patch is registered in data_types.py
    surface = Surface.load(str(file_path))
    xdim = float(surface.step_x * micro)  # Convert from µm to m
    ydim = float(surface.step_y * micro)  # Convert from µm to m

    # Extract metadata if available from our patched reader
    metadata = getattr(surface, "metadata", None) or {}
    header = metadata.get("header", {})
    texture_data = metadata.get("texture_data")
    quality_data = metadata.get("quality_data")
    vertical_resolution = metadata.get("vr")
    lateral_resolution = metadata.get("lr")
    xml_data = metadata.get("XMLData")

    # Try to extract VR/LR from XML data if not already extracted
    if not (vertical_resolution and lateral_resolution) and xml_data:
        extracted_lr, extracted_vr = extract_resolutions_from_xml_data(xml_data)
        lateral_resolution = lateral_resolution or extracted_lr
        vertical_resolution = vertical_resolution or extracted_vr

    return ImageData(
        type=ImageType.SURFACE,
        mark_type="",  # MATLAB: data_out.mark_type = ''
        depth_data=np.asarray(surface.data, dtype=np.float64),
        texture_data=texture_data,
        quality_data=quality_data,
        xdim=xdim,
        ydim=ydim,
        xdim_orig=xdim,
        ydim_orig=ydim,
        invalid_pixel_val=np.nan,  # MATLAB: converted invalid pixels to NaN
        vertical_resolution=vertical_resolution,
        lateral_resolution=lateral_resolution,
        input_format=SupportedExtension.AL3D,
        additional_info={
            "Header": header,
            "XMLData": xml_data,
        },
        subsampling=1,
        orig_path=str(file_path),
    )
