"""AL3D file format parser.

AL3D is Alicona's proprietary format for storing surface topography data.
This parser translates the MATLAB AliconaReader functionality to Python.
"""

from pathlib import Path
from typing import Any

import numpy as np
from surfalize import Surface

from models.enums import ImageType, SupportedExtension
from models.image import ImageData


def _extract_resolution_from_description(
    description_text: str, resolution_type: str
) -> float | None:
    """
    Extract vertical or lateral resolution from XML description text.

    This function replicates the MATLAB logic from importdata.m (lines 279-359)
    for parsing resolution values and their units from Alicona XML metadata.

    Parameters
    ----------
    description_text : str
        The XML description text containing resolution information
    resolution_type : str
        Either 'Vertical' or 'Lateral' to specify which resolution to extract

    Returns
    -------
    float | None
        Resolution value in meters, or None if not found

    Notes
    -----
    The MATLAB code uses ASCII/Unicode character codes to identify units:
    - m=109, n=110, µ=181
    This is necessary because the µ character can have different encodings.
    """
    if not description_text:
        return None

    try:
        search_string = f"Estimated {resolution_type} Resolution:"

        # Find the start of the resolution value
        start_idx = description_text.find(search_string)
        if start_idx == -1:
            return None

        start_idx += len(search_string)

        # Skip whitespace (ASCII 32) - MATLAB: while uint8(text(start)) == 32
        while (
            start_idx < len(description_text) and ord(description_text[start_idx]) == 32
        ):
            start_idx += 1

        # Find end of line (CR=13 or LF=10)
        # MATLAB: while uint8(text(end)) ~= 13 && uint8(text(end)) ~= 10
        end_idx = start_idx
        while end_idx < len(description_text):
            char_code = ord(description_text[end_idx])
            if char_code == 13 or char_code == 10:  # CR or LF
                break
            end_idx += 1

        resolution_str = description_text[start_idx:end_idx]

        # Extract numeric value
        # MATLAB: tmp = uint8(text) == 46 | (uint8(text) >= 48 & uint8(text) <= 57)
        # This extracts digits (48-57) and decimal point (46)
        numeric_chars = [
            c for c in resolution_str if ord(c) == 46 or (48 <= ord(c) <= 57)
        ]
        if not numeric_chars:
            return None
        value = float("".join(numeric_chars))

        # Extract unit based on character codes
        # MATLAB: unit = uint8(text) == 109 | uint8(text) == 110 | uint8(text) == 181
        unit_chars = [ord(c) for c in resolution_str if ord(c) in [109, 110, 181]]

        if not unit_chars:
            return None

        # Determine unit signature and convert to meters
        # MATLAB uses switch/case on char() of the unit codes
        unit_signature = tuple(unit_chars)

        match unit_signature:
            case (109,):  # char(109) = 'm'
                return value
            case (109, 109):  # char([109 109]) = 'mm'
                return value * 1e-3
            case (181, 109):  # char([181 109]) = 'µm'
                return value * 1e-6
            case (110, 109):  # char([110 109]) = 'nm'
                return value * 1e-9
            case _:
                return None

    except (IndexError, ValueError):
        return None


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
    xdim = float(surface.step_x * 1e-6)  # Convert from µm to m
    ydim = float(surface.step_y * 1e-6)  # Convert from µm to m

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
        # Try to read XML data from the file
        try:
            # Read the file to get XML data
            with open(file_path, "rb") as f:
                # Read header to find XMLOffset and XMLSize
                from surfalize.file.al3d import MAGIC, read_tag

                magic = f.read(17)
                if magic == MAGIC:
                    header_dict = {}
                    key, value = read_tag(f)
                    if key == "Version":
                        header_dict[key] = value

                    key, value = read_tag(f)
                    if key == "TagCount":
                        header_dict[key] = value
                        tag_count = int(value)

                        for _ in range(tag_count):
                            key, value = read_tag(f)
                            header_dict[key] = value

                        # Extract XML if available
                        if "XMLOffset" in header_dict and "XMLSize" in header_dict:
                            xml_offset = int(header_dict["XMLOffset"])
                            xml_size = int(header_dict["XMLSize"])

                            if xml_offset > 0 and xml_size > 0:
                                f.seek(xml_offset)
                                xml_bytes = f.read(xml_size)
                                xml_text = xml_bytes.decode("utf-8", errors="ignore")

                                # MATLAB: start_LR = strfind(..., 'Estimated Lateral Resolution:')
                                # MATLAB: start_VR = strfind(..., 'Estimated Vertical Resolution:')
                                vertical_resolution = (
                                    _extract_resolution_from_description(
                                        xml_text, "Vertical"
                                    )
                                )
                                lateral_resolution = (
                                    _extract_resolution_from_description(
                                        xml_text, "Lateral"
                                    )
                                )

                                # Store XML data in additional_info
                                xml_data = {"raw_xml": xml_text}

        except Exception:
            # If XML extraction fails, VR and LR remain None
            # MATLAB: catch block sets VR = [], LR = []
            pass

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
        crop_type=None,
        crop_coordinates=None,
        crop_info=[],
        cutoff_hi=[],
        cutoff_lo=[],
        is_prep=False,
        is_crop=False,
        is_interp=False,
        is_resamp=False,
        data_param={},
        subsampling=1,
        orig_path=str(file_path),
    )
