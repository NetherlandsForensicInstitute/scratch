"""AL3D file format parser.

AL3D is Alicona's proprietary format for storing surface topography data.
This parser translates the MATLAB AliconaReader functionality to Python.
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import micro
from surfalize import Surface

from models.enums import ImageType, SupportedExtension
from models.image import ImageData


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

    Notes
    -----
    MATLAB equivalent (importdata.m, lines 256-380):
    - Reads depth_data, texture_data, quality_data
    - Extracts pixel sizes from header (PixelSizeXMeter, PixelSizeYMeter)
    - Converts invalid pixels (> 1e9) to NaN
    - Extracts VR/LR from XMLData if available
    - Performs unit conversion for VR/LR

    The surfalize library with our patched reader handles the basic file reading
    and invalid pixel conversion. This function focuses on metadata extraction.
    """
    # Load the surface using surfalize with our patched reader
    # The patch is registered in data_types.py
    surface = Surface.load(str(file_path))

    # Extract basic data
    # MATLAB: depth_data = double(data.DepthData)
    depth_data = np.asarray(surface.data, dtype=np.float64)

    # MATLAB: xdim = str2double(data.Header.PixelSizeXMeter)
    # MATLAB: ydim = str2double(data.Header.PixelSizeYMeter)
    # Note: surfalize already extracts these and stores them in step_x, step_y
    # They are in meters in the original file
    xdim = float(surface.step_x * micro)  # Convert from µm to m
    ydim = float(surface.step_y * micro)  # Convert from µm to m

    # Initialize metadata fields
    texture_data = None
    quality_data = None
    vertical_resolution = None
    lateral_resolution = None
    header = {}
    xml_data = None

    # Extract metadata if available from our patched reader
    if hasattr(surface, "metadata") and surface.metadata:
        metadata = surface.metadata
        header = metadata.get("header", {})
        texture_data = metadata.get("texture_data")
        quality_data = metadata.get("quality_data")
        vertical_resolution = metadata.get("vr")
        lateral_resolution = metadata.get("lr")

    # Try to extract VR/LR from XML data if not already extracted
    # MATLAB: if isfield(data, 'XMLData')
    if vertical_resolution is None or lateral_resolution is None:
        pass  # TODO: redo this section

    # Build additional_info matching MATLAB structure
    # MATLAB: additional_info.Header = data.Header
    # MATLAB: if isfield(data, 'XMLData'), additional_info.XMLData = data.XMLData
    additional_info: dict[str, Any] = {
        "Header": header,
    }

    if xml_data:
        additional_info["XMLData"] = xml_data

    # Return ImageData matching MATLAB data_out structure
    # MATLAB: data_out.type = 'surface'
    return ImageData(
        type=ImageType.SURFACE,
        mark_type="",  # MATLAB: data_out.mark_type = ''
        depth_data=depth_data,
        texture_data=texture_data,
        quality_data=quality_data,
        xdim=xdim,
        ydim=ydim,
        xdim_orig=xdim,  # Not modified during loading
        ydim_orig=ydim,
        invalid_pixel_val=np.nan,  # MATLAB: converted invalid pixels to NaN
        vertical_resolution=vertical_resolution,  # MATLAB: VR
        lateral_resolution=lateral_resolution,  # MATLAB: LR
        input_format=SupportedExtension.AL3D,
        additional_info=additional_info,
        subsampling=1,
        orig_path=str(file_path),
    )
