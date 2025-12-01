from functools import partial
from typing import Optional, Union, Callable

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
    downweight_outliers: bool = False,
    outlier_tol: float = 1e-2,
    n_downweight_outlier_iter: int = 50,
) -> np.ndarray:
    """Apply kernel filter with optional weighting, regression, and outlier downweighting.

    :param kernel_col: Column kernel, 2D kernel, or tuple (col_kernel, row_kernel).
    :param kernel_row: Row kernel (optional, ignored if kernel_col is tuple).
    :param data: Input data.
    :param weights: Weight array (optional).
    :param regression_order: Regression order (0, 1, or 2).
    :param nan_out: Return original NaN values as NaN.
    :param downweight_outliers: Use iterative biweight to downweight outliers.
    :param outlier_tol: Convergence tolerance for outlier downweighting.
    :param n_downweight_outlier_iter: Maximum iterations for outlier downweighting.
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

    if not downweight_outliers:
        return filtered

    # Apply iterative biweight to downweight outliers
    filter_func = partial(
        _filter_kernel_single, kernel_col, kernel_row, regression_order=regression_order
    )

    return _biweight_refine(
        filter_func, data, filtered, outlier_tol, n_downweight_outlier_iter, nan_out
    )


def _filter_kernel_single(
    kernel_col: Optional[Union[np.ndarray, tuple[np.ndarray, np.ndarray]]],
    kernel_row: Optional[np.ndarray],
    data: np.ndarray,
    weights: Optional[np.ndarray] = None,
    regression_order: int = 0,
    nan_out: bool = True,
) -> np.ndarray:
    """Apply single pass of kernel filter (used internally by _biweight_refine).

    :param kernel_col: Column kernel, 2D kernel, or tuple (col_kernel, row_kernel).
    :param kernel_row: Row kernel (optional, ignored if kernel_col is tuple).
    :param data: Input data.
    :param weights: Weight array (optional).
    :param regression_order: Regression order (0, 1, or 2).
    :param nan_out: Return original NaN values as NaN.
    :return: Filtered data.
    """
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


def _biweight_refine(
    filter_func: Callable,
    data: np.ndarray,
    initial_filtered: np.ndarray,
    outlier_tol: float,
    n_downweight_outlier_iter: int,
    nan_out: bool,
    **filter_kwargs,
) -> np.ndarray:
    """Refine filtering using Tukey's biweight to downweight outliers.

    :param filter_func: Filtering function to call iteratively.
    :param data: Original input data.
    :param initial_filtered: Initial filtered result.
    :param outlier_tol: Convergence tolerance (relative to mean).
    :param n_downweight_outlier_iter: Maximum iterations.
    :param nan_out: Return NaN values as NaN.
    :param filter_kwargs: Additional arguments for filter_func.
    :return: Filtered data with outliers downweighted.
    """
    # Tukey's biweight parameters:
    # - BIWEIGHT_C: cutoff in sigma units (3/sqrt(2) ≈ 2.12, tighter than standard 4.685)
    # - MAD_TO_SIGMA: converts MAD to sigma estimate (1/0.6745 ≈ 1.48 for normal distribution)
    #
    # The cutoff for zero weight is: cutoff = BIWEIGHT_C * sigma = BIWEIGHT_C * MAD_TO_SIGMA * MAD
    # This equals the original: r_scale * MAD = (3 / (sqrt(2) * 0.6745)) * MAD
    # Since: BIWEIGHT_C * MAD_TO_SIGMA = (3/sqrt(2)) * (1/0.6745) = 3 / (sqrt(2) * 0.6745) = r_scale
    BIWEIGHT_C = 3 / np.sqrt(2)
    MAD_TO_SIGMA = 1 / 0.6745
    MIN_WEIGHT = 1e-3

    # Early exit if initial_filtered is zero or all NaN
    mean_abs_filtered = np.nanmean(np.abs(initial_filtered))
    if mean_abs_filtered == 0 or np.isnan(mean_abs_filtered):
        return initial_filtered
    conv_tol = outlier_tol * mean_abs_filtered

    # Precompute NaN mask (doesn't change between iterations)
    nan_mask = np.isnan(data)

    filtered = initial_filtered
    filtered_old = data  # First iteration: compare against original data

    for iteration in range(1, n_downweight_outlier_iter + 1):
        # Check convergence (skip first iteration)
        if iteration > 1:
            mac = np.nanmean(np.abs(filtered - filtered_old))
            if mac <= conv_tol:
                break

        # Swap references instead of copying (filter_func returns a new array)
        filtered_old = filtered

        # Calculate residuals and robust scale estimate
        residuals = data - filtered
        abs_residuals = np.abs(residuals)
        sigma = MAD_TO_SIGMA * np.nanmedian(abs_residuals)

        # Use larger scale on first iteration for stability
        if iteration == 1:
            sigma *= 2

        # Ensure minimum scale to avoid division by zero
        sigma = max(sigma, 0.1 * np.nanstd(residuals), np.finfo(float).eps)

        # Tukey's biweight weights: w(u) = (1 - u^2)^2 for |u| <= 1, else 0
        cutoff = BIWEIGHT_C * sigma
        u = residuals / cutoff
        data_weights = np.where(abs_residuals <= cutoff, (1 - u**2) ** 2, 0.0)
        data_weights = np.clip(data_weights, MIN_WEIGHT, 1.0)
        data_weights[nan_mask] = 0

        # Filter with updated weights
        filtered = filter_func(
            data=data, weights=data_weights, nan_out=nan_out, **filter_kwargs
        )

    return filtered
