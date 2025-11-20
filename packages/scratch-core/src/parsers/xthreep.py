"""X3P file format parser.

X3P (ISO 25178-72) is an XML-based standard format for representing
surface texture and other surface data.
"""

from pathlib import Path
from typing import Any

import numpy as np

from models.enums import ImageType
from models.image import ImageData
from x3p import X3Pfile


def _calculate_dimension(
    coord_array: np.ndarray,
    primary_row: int,
    primary_col: int,
    fallback_row: int,
    fallback_col: int,
) -> float:
    """Calculate dimension from coordinate array differences.

    Parameters
    ----------
    coord_array : np.ndarray
        Coordinate array (x or y)
    primary_row : int
        Primary row index to try for dimension calculation
    primary_col : int
        Primary column index to try for dimension calculation
    fallback_row : int
        Fallback row index if primary calculation yields 0
    fallback_col : int
        Fallback column index if primary calculation yields 0

    Returns
    -------
    float
        Calculated dimension value in meters

    Notes
    -----
    MATLAB uses 1-based indexing, Python uses 0-based:
    - MATLAB y(2,1) -> Python y[1,0]
    - MATLAB y(1,2) -> Python y[0,1]
    """
    dim = coord_array[primary_row, primary_col] - coord_array[0, 0]
    if dim == 0:
        dim = coord_array[fallback_row, fallback_col] - coord_array[0, 0]
    return float(dim)


def load_x3p_file(file_path: Path) -> ImageData:
    """
    Load X3P file format (ISO 25178-72).

    Parameters
    ----------
    file_path : Path
        Path to the .x3p file

    Returns
    -------
    ImageData
        Loaded data structure

    Raises
    ------
    ValueError
        If required fields are missing or invalid

    Notes
    -----
    X3P files can contain either surface data (2D) or profile data (1D).
    The format uses ITK coordinate system which may require transposition.

    For surfaces:
        - xdim = y(2,1) - y(1,1) [fallback: y(1,2) - y(1,1)]
        - ydim = x(1,2) - x(1,1) [fallback: x(2,1) - x(1,1)]

    For profiles:
        - xdim = x(2,1) - x(1,1) [fallback: y(2,1) - y(1,1)]
        - ydim = None
    """

    # Read the X3P file
    x3p_data = X3Pfile(str(file_path))

    # Get the measurement data (masked numpy array)
    z = x3p_data.data

    # Get feature type from record1
    feature_type = getattr(x3p_data.record1, "featuretype", "").upper()

    # Build coordinate arrays from axis information
    # pyx3p stores axis info in record1.axes.CX, CY, CZ
    cx_axis = x3p_data.record1.axes.CX
    cy_axis = x3p_data.record1.axes.CY

    # Calculate coordinate grids based on increment and offset
    # X3P format: offset + (index * increment)
    cx_increment = float(getattr(cx_axis, "increment", None) or 0)
    cy_increment = float(getattr(cy_axis, "increment", None) or 0)
    cx_offset = float(getattr(cx_axis, "offset", None) or 0)
    cy_offset = float(getattr(cy_axis, "offset", None) or 0)

    # Create coordinate arrays
    # For a 2D surface, we need the grid of x,y coordinates
    if z.ndim == 3:
        # Multi-layer data: shape is [layers, y_dim, x_dim]
        z = z[0]  # Take first layer

    ny, nx = z.shape if z.ndim == 2 else (1, len(z))

    # Build coordinate grids matching MATLAB's convention
    x = np.zeros((ny, nx))
    y = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            x[i, j] = cx_offset + j * cx_increment
            y[i, j] = cy_offset + i * cy_increment

    # Store processing info and metadata
    pinfo = x3p_data.record1
    meta = getattr(x3p_data, "record2", None)

    # Handle depth_data transposition
    # MATLAB: if size(z, 2) > 1 (if z has more than 1 column)
    if z.ndim == 2 and z.shape[1] > 1:
        depth_data = z.T  # Transpose for 2D data
    else:
        depth_data = z

    # Texture and quality data are not present in basic X3P files
    texture_data = None
    quality_data = None

    # Determine type based on FeatureType (already obtained on line 98)
    if feature_type == "SUR":
        # Surface data
        image_type = ImageType.SURFACE

        # Calculate dimensions using ITK coordinate system
        # MATLAB: xdim = y(2,1) - y(1,1), with fallback to y(1,2) - y(1,1)
        # Python: y[1,0] - y[0,0], with fallback to y[0,1] - y[0,0]
        xdim = _calculate_dimension(y, 1, 0, 0, 1)

        # MATLAB: ydim = x(1,2) - x(1,1), with fallback to x(2,1) - x(1,1)
        # Python: x[0,1] - x[0,0], with fallback to x[1,0] - x[0,0]
        ydim = _calculate_dimension(x, 0, 1, 1, 0)

    else:
        # Profile data
        image_type = ImageType.PROFILE

        # MATLAB: xdim = x(2,1) - x(1,1), with fallback to y(2,1) - y(1,1)
        # Python: x[1,0] - x[0,0], with fallback to y[1,0] - y[0,0]
        xdim = float(x[1, 0] - x[0, 0])
        if xdim == 0:
            xdim = float(y[1, 0] - y[0, 0])

        ydim = 0.0  # No ydim for profiles

    # Vertical and lateral resolution
    # VR = [] (not calculated)
    # LR = [] (commented out: LR = 2 * xdim)
    vertical_resolution = None
    lateral_resolution = None

    # Store additional info
    additional_info: dict[str, Any] = {
        "pinfo": pinfo,
        "meta": meta,
    }

    return ImageData(
        type=image_type,
        depth_data=depth_data,
        texture_data=texture_data,
        quality_data=quality_data,
        xdim=xdim,
        ydim=ydim,
        vertical_resolution=vertical_resolution,
        lateral_resolution=lateral_resolution,
        additional_info=additional_info,
    )
