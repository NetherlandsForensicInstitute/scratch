import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


def _create_kernel_1d(
    size: int, cutoff_pixel: float, alpha: float
) -> NDArray[np.floating]:
    """Create a 1D Gaussian kernel."""
    radius = (size - 1) // 2
    vp = np.arange(-radius, radius + 1)
    return np.exp(-np.pi * (vp / (alpha * cutoff_pixel)) ** 2) / (alpha * cutoff_pixel)


def _apply_r0_filter(
    alpha: float,
    cutoff_pixels: NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply order-0 regression filter (weighted Gaussian average)."""
    kernel_x, kernel_y = _get_normalized_kernels(alpha, cutoff_pixels)

    # Weighted filtering for NaN handling
    weights = np.where(np.isnan(data), 0.0, 1.0)
    data_filled = np.where(np.isnan(data), 0.0, data)

    filtered = _convolve_separable(data_filled * weights, kernel_x, kernel_y)
    weight_sum = _convolve_separable(weights, kernel_x, kernel_y)

    with np.errstate(invalid="ignore", divide="ignore"):
        return filtered / weight_sum


def _convolve_separable(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply separable convolution using 1D kernels."""
    result = fftconvolve(data, kernel_y[:, np.newaxis], mode="same")
    return fftconvolve(result, kernel_x[np.newaxis, :], mode="same")


def _apply_r12_filter(
    alpha: float,
    cutoff_pixels: NDArray[np.floating],
    data: NDArray[np.floating],
    regression_order: int,
) -> NDArray:
    """Apply order-1 or order-2 regression filter (local polynomial fitting)."""
    kernel_x, kernel_y = _get_normalized_kernels(alpha, cutoff_pixels)

    ny, nx = len(kernel_y), len(kernel_x)
    radius_y, radius_x = (ny - 1) // 2, (nx - 1) // 2

    # Normalized coordinate vectors
    vpi = np.arange(-radius_y, radius_y + 1) / radius_y if ny > 1 else np.array([0])
    vpj = np.arange(-radius_x, radius_x + 1) / radius_x if nx > 1 else np.array([0])

    # Generate polynomial terms where sum of powers <= order
    noi = min(data.shape[0] - 1, regression_order, ny - 1)
    noj = min(data.shape[1] - 1, regression_order, nx - 1)
    mij = np.array(
        [
            (pi, pj)
            for pi in range(noi + 1)
            for pj in range(noj + 1)
            if pi + pj <= regression_order
        ]
    )
    n_params = len(mij)

    # Unique terms for design matrix
    mij_A = np.array(
        [
            (mij[p, 0] + mij[q, 0], mij[p, 1] + mij[q, 1])
            for p in range(n_params)
            for q in range(n_params)
        ]
    )
    mij_A_unique, inverse_indices = np.unique(mij_A, axis=0, return_inverse=True)

    # Prepare weighted data
    weights = np.where(np.isnan(data), 0.0, 1.0)
    weighted_data = np.where(np.isnan(data), 0.0, data * weights)

    # Compute filtered terms for c vector (data side)
    mc = np.array(
        [
            _convolve_separable(
                weighted_data, (vpj**pj) * kernel_x, (vpi**pi) * kernel_y
            )
            for pi, pj in mij
        ]
    )

    # Compute filtered terms for A matrix (weights side)
    mA_unique = np.array(
        [
            _convolve_separable(weights, (vpj**pj) * kernel_x, (vpi**pi) * kernel_y)
            for pi, pj in mij_A_unique
        ]
    )
    mA = mA_unique[inverse_indices]

    # Reshape for batch solving: (n_params, n_params, ny, nx) -> (ny, nx, n_params, n_params)
    mA = np.moveaxis(mA.reshape(n_params, n_params, *data.shape), [0, 1], [-2, -1])
    mc = np.moveaxis(mc, 0, -1)[..., np.newaxis]

    # Solve linear system at all pixels
    result = np.full(data.shape, np.nan)
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)

    A_valid = mA[valid_indices]
    c_valid = mc[valid_indices]

    # Try batch solve first
    try:
        solutions = np.linalg.solve(A_valid, c_valid)
        result[valid_indices] = solutions[:, 0, 0]
    except np.linalg.LinAlgError:
        # Fall back to per-pixel lstsq for robustness
        for i in range(len(valid_indices[0])):
            idx = (valid_indices[0][i], valid_indices[1][i])
            result[idx] = np.linalg.lstsq(mA[idx], mc[idx], rcond=None)[0][0, 0]

    return result


def _get_normalized_kernels(
    alpha: float, cutoff_pixels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Create normalized 1D Gaussian kernels for separable convolution."""
    kernel_size = 1 + np.ceil(len(cutoff_pixels) * cutoff_pixels).astype(int)
    kernel_size = kernel_size + (1 - kernel_size % 2)

    kernel_y = _create_kernel_1d(kernel_size[0], cutoff_pixels[0], alpha)
    kernel_x = _create_kernel_1d(kernel_size[1], cutoff_pixels[1], alpha)

    # Normalize kernels
    total = np.sum(kernel_y) * np.sum(kernel_x)
    kernel_y, kernel_x = kernel_y / np.sqrt(total), kernel_x / np.sqrt(total)
    return kernel_x, kernel_y


def apply_gaussian_filter(
    data: NDArray[np.floating],
    cutoff_length: float,
    pixel_size: tuple[float, float] = (1.0, 1.0),
    regression_order: int = 0,
    nan_out: bool = True,
    is_high_pass: bool = False,
) -> NDArray:
    """
    Apply Gaussian filter with optional polynomial regression.

    :param data: Input 2D data array.
    :param cutoff_length: Cutoff wavelength in physical units
    :param pixel_size: Pixel size in physical units (y, x).
    :param regression_order: Polynomial regression order (0, 1, or 2).
    :param nan_out: Whether to preserve NaN positions in output.
    :param is_high_pass: Whether to return high-pass result (data - filtered).
    :return: Filtered data array.
    """
    cutoff_pixels = cutoff_length / np.array(pixel_size)
    alpha = np.sqrt(np.log(2) / np.pi) if regression_order < 2 else 0.7309134280946760

    if regression_order == 0:
        filtered = _apply_r0_filter(alpha, cutoff_pixels, data)
    else:
        filtered = _apply_r12_filter(alpha, cutoff_pixels, data, regression_order)

    if nan_out:
        filtered[np.isnan(data)] = np.nan

    return data - filtered if is_high_pass else filtered
