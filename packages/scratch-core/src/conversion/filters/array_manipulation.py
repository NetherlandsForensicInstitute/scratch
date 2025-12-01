from typing import Tuple

import numpy as np

from conversion.filters.data_formats import (
    TrimType,
    Trim1D,
    Trim2D,
    PadType,
    Pad1D,
    Pad2D,
)


def crop_array(data: np.ndarray, trim: TrimType) -> np.ndarray:
    """Crop array by removing rows/columns from borders.

    :param data: Input array (1D or 2D).
    :param trim: Trim1D or Trim2D object specifying how much to remove (must be >= 0).
    :return: Cropped array.
    :raises ValueError: If invalid dimensions or trim type mismatch.
    """
    data = np.asarray(data)
    if data.ndim > 2:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D")

    if data.ndim == 1:
        if not isinstance(trim, Trim1D):
            raise ValueError(f"Expected Trim1D for 1D array, got {type(trim)}")
        return data[trim.start : data.size - trim.end]
    else:
        if not isinstance(trim, Trim2D):
            raise ValueError(f"Expected Trim2D for 2D array, got {type(trim)}")
        return data[
            trim.top : data.shape[0] - trim.bottom,
            trim.left : data.shape[1] - trim.right,
        ]


def pad_array(data: np.ndarray, pad: PadType) -> np.ndarray:
    """Pad array by adding NaN-filled rows/columns to borders.

    :param data: Input array (1D or 2D).
    :param pad: Pad1D or Pad2D object specifying how much to pad (must be >= 0).
    :return: Padded array (converted to float if necessary to support NaN).
    :raises ValueError: If invalid dimensions, pad type mismatch, or negative pad values.
    """
    data = np.asarray(data)
    if data.ndim > 2:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D")

    # Convert to float if integer type (needed for NaN)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)

    if data.ndim == 1:
        if not isinstance(pad, Pad1D):
            raise ValueError(f"Expected Pad1D for 1D array, got {type(pad)}")
        if pad.start < 0 or pad.end < 0:
            raise ValueError(
                f"Pad values must be non-negative, got start={pad.start}, end={pad.end}"
            )
        return np.pad(data, (pad.start, pad.end), constant_values=np.nan)
    else:
        if not isinstance(pad, Pad2D):
            raise ValueError(f"Expected Pad2D for 2D array, got {type(pad)}")
        if pad.top < 0 or pad.bottom < 0 or pad.left < 0 or pad.right < 0:
            raise ValueError(
                f"Pad values must be non-negative, got "
                f"top={pad.top}, bottom={pad.bottom}, left={pad.left}, right={pad.right}"
            )
        return np.pad(
            data, ((pad.top, pad.bottom), (pad.left, pad.right)), constant_values=np.nan
        )


def calculate_nan_trim(data: np.ndarray) -> TrimType:
    """Determine trim values for removing NaN borders.

    :param data: Input array (1D or 2D).
    :return: Trim1D for 1D arrays or Trim2D for 2D arrays.
    :raises ValueError: If invalid dimensions.
    """
    data = np.asarray(data)
    if data.ndim > 2:
        raise ValueError(f"Data must be 1D or 2D, got {data.ndim}D")

    # Handle 1D case
    if data.ndim == 1:
        valid_mask = ~np.isnan(data)
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) == 0:
            return Trim1D(start=0, end=data.size)

        return Trim1D(start=valid_idx[0], end=data.size - valid_idx[-1] - 1)

    # Handle 2D case
    valid_mask = ~np.isnan(data)

    # Find rows and columns with valid data
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    row_idx = np.where(valid_rows)[0]
    col_idx = np.where(valid_cols)[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        return Trim2D(top=0, bottom=data.shape[0], left=0, right=data.shape[1])

    return Trim2D(
        top=row_idx[0],
        bottom=data.shape[0] - row_idx[-1] - 1,
        left=col_idx[0],
        right=data.shape[1] - col_idx[-1] - 1,
    )


def crop_nan_borders(data: np.ndarray) -> Tuple[np.ndarray, TrimType]:
    """Crop array by removing NaN border rows and columns.

    :param data: Input array.
    :return: Tuple of (cropped_array, trim_amounts).
    """
    trim = calculate_nan_trim(data)
    cropped = crop_array(data, trim)
    return cropped, trim
