from typing import Optional, Union

import numpy as np
from scipy import ndimage

from conversion.filters.data_formats import FilterDomain
from conversion.filters.utils import _init_nan_weights
from conversion.filters.validation import _validate_domain


def create_footprint(size: np.ndarray, domain: FilterDomain) -> np.ndarray:
    """Create kernel domain mask (rectangle or disk).

    :param size: Size of kernel [rows, cols].
    :param domain: FilterDomain enum (DISK or RECTANGLE).
    :return: Boolean mask for kernel domain.
    :raises ValueError: If domain is invalid or size is invalid.
    """
    size = np.asarray(size, dtype=int)
    _validate_domain(domain, (FilterDomain.DISK, FilterDomain.RECTANGLE))

    # Validate size
    if np.any(size < 1):
        raise ValueError(f"Kernel size must be >= 1, got {size}")

    # Rectangle for explicit rectangle domain or 1D kernels
    if domain == FilterDomain.RECTANGLE or np.any(size == 1):
        return np.ones(size, dtype=bool)

    # Disk/ellipse domain
    radius = np.ceil((size - 1) / 2).astype(int)
    radius_mask = size / 2
    y, x = np.ogrid[-radius[0] : radius[0] + 1, -radius[1] : radius[1] + 1]
    return (y / radius_mask[0]) ** 2 + (x / radius_mask[1]) ** 2 <= 1


def create_averaging_kernel(size: np.ndarray) -> np.ndarray:
    """Create disk-shaped averaging kernel.

    :param size: Kernel size [rows, cols].
    :return: Normalized averaging kernel with disk domain.
    """
    size = np.asarray(size)

    # Create disk domain
    footprint = create_footprint(size, FilterDomain.DISK)
    size = footprint.shape

    # Set kernel values
    kernel = np.zeros(size)
    kernel[footprint] = 1.0 / np.sum(footprint)

    return kernel


def filter_kernel(
    kernel_col: Optional[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]],
    kernel_row: Optional[np.ndarray],
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regression_order: int = 0,
    nan_out: bool = True,
) -> np.ndarray:
    """Apply kernel filter with optional weighting and regression.

    :param kernel_col: Column kernel, 2D kernel, or tuple (col_kernel, row_kernel).
    :param kernel_row: Row kernel (optional, ignored if kernel_col is tuple).
    :param data: Input data.
    :param weights: Weight array (optional).
    :param regression_order: Regression order (0, 1, or 2).
    :param nan_out: Return original NaN values as NaN.
    :return: Filtered data.
    """
    if regression_order not in [0, 1, 2]:
        raise ValueError("Invalid regression order")

    if regression_order > 0:
        raise NotImplementedError(
            "Regression filtering for order > 0 not yet implemented."
        )

    if weights is None:
        weights = _init_nan_weights(data)

    data_weighted = np.where(np.isnan(data), 0, data * weights)

    if isinstance(kernel_col, tuple):
        # Separable kernel passed as tuple
        col_kernel = kernel_col[0]
        row_kernel = kernel_col[1]
        filtered = ndimage.convolve1d(
            data_weighted, col_kernel.flatten(), axis=0, mode="constant", cval=0.0
        )
        filtered = ndimage.convolve1d(
            filtered, row_kernel.flatten(), axis=1, mode="constant", cval=0.0
        )
        weight_sum = ndimage.convolve1d(
            weights, col_kernel.flatten(), axis=0, mode="constant", cval=0.0
        )
        weight_sum = ndimage.convolve1d(
            weight_sum, row_kernel.flatten(), axis=1, mode="constant", cval=0.0
        )
    elif kernel_row is not None and kernel_row.size > 1:
        # Separable kernel passed as separate col/row arguments
        assert kernel_col is not None
        filtered = ndimage.convolve1d(
            data_weighted, kernel_col.flatten(), axis=0, mode="constant", cval=0.0
        )
        filtered = ndimage.convolve1d(
            filtered, kernel_row.flatten(), axis=1, mode="constant", cval=0.0
        )
        weight_sum = ndimage.convolve1d(
            weights, kernel_col.flatten(), axis=0, mode="constant", cval=0.0
        )
        weight_sum = ndimage.convolve1d(
            weight_sum, kernel_row.flatten(), axis=1, mode="constant", cval=0.0
        )
    else:
        # 2D kernel
        kernel = kernel_col if kernel_col is not None else np.array([[1.0]])
        filtered = ndimage.convolve(data_weighted, kernel, mode="constant", cval=0.0)
        weight_sum = ndimage.convolve(weights, kernel, mode="constant", cval=0.0)

    weight_sum[weight_sum == 0] = np.nan
    filtered = filtered / weight_sum

    if nan_out:
        filtered[np.isnan(data)] = np.nan

    return filtered
