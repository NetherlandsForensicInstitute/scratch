import numpy as np
from conversion.filter import (
    _convolve_2d_separable,
    _build_lhs_matrix,
    _solve_pixelwise_regression,
)
from numpy.typing import NDArray
from conversion.filter import (
    _apply_order0_filter,
    _create_normalized_separable_kernels,
)

from renders.computations import generate_polynomial_exponents
from utils.constants import RegressionOrder

# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
# alpha = Sqrt((-1 - LambertW(-1, -1 / (2 * exp(1)))) / Pi)
ALPHA_REGRESSION = 0.7309134280946760


def calculate_polynomial_filter(
    data: NDArray[np.floating],
    kernel_x: NDArray[np.floating],
    kernel_y: NDArray[np.floating],
    exponents: list[tuple[int, int]],
) -> NDArray[np.floating]:
    """
    Apply Order-1 or Order-2 Local Polynomial Regression.

    This function performs a Weighted Least Squares (WLS) fit of a polynomial surface within a local window
    defined by the kernels. For each pixel, it solves the linear system A * c = b, where 'c' contains the
    coefficients of the polynomial. The smoothed value is the first coefficient (c0).

    The kernels (kernel_x, kernel_y) serve as spatial weight functions. They determine the importance of
    neighboring pixels in the regression. A non-uniform kernel (e.g., Gaussian) ensures that points closer
    to the target pixel have a higher influence on the fit than points at the window's edge, providing better
    localization and noise suppression.

    :param data: The 2D input array to be filtered. Can contain NaNs, which are treated as zero-weight during
        the regression.
    :param kernel_x: 1D array representing the horizontal weight distribution.
    :param kernel_y: 1D array representing the vertical weight distribution.
    :param exponents: List of (power_y, power_x) tuples defining the polynomial terms.
    :returns: The filtered (smoothed) version of the input data.
    """
    # 1. Setup Coordinate Systems (Normalized to [-1, 1] for stability)
    ny, nx = len(kernel_y), len(kernel_x)
    radius_y, radius_x = (ny - 1) // 2, (nx - 1) // 2

    y_coords = np.arange(-radius_y, radius_y + 1) / (radius_y if ny > 1 else 1.0)
    x_coords = np.arange(-radius_x, radius_x + 1) / (radius_x if nx > 1 else 1.0)

    # 2. Construct the Linear System Components (A matrix and b vector)
    nan_mask = np.isnan(data)
    weights = np.where(nan_mask, 0.0, 1.0)
    weighted_data = np.where(nan_mask, 0.0, data * weights)

    # Calculate RHS vector 'b' (Data Moments)
    # b_k = Convolution(weighted_data, x^px * y^py * Kernel)
    rhs_moments = np.array(
        [
            _convolve_2d_separable(
                weighted_data, (x_coords**px) * kernel_x, (y_coords**py) * kernel_y
            )
            for py, px in exponents
        ]
    )

    # Calculate LHS Matrix 'A' (Weight Moments)
    # A_jk = Convolution(weights, x^(px_j + px_k) * y^(py_j + py_k) * Kernel)
    lhs_matrix = _build_lhs_matrix(
        weights, kernel_x, kernel_y, x_coords, y_coords, exponents
    )

    # 3. Solve the System (A * c = b) per pixel
    return _solve_pixelwise_regression(lhs_matrix, rhs_moments, data)


def apply_gaussian_regression_filter(
    data: NDArray[np.floating],
    cutoff_pixels: NDArray[np.floating],
    regression_order: RegressionOrder,
) -> NDArray[np.floating]:
    """
    Apply a 2D Savitzky-Golay filter with Gaussian weighting via local polynomial regression (ISO 16610-21).

    This implementation generalizes standard Gaussian filtering to handle missing data (NaNs) using local
    regression techniques. It supports 0th order (Gaussian Kernel weighted average), 1st order (planar fit),
    and 2nd order (quadratic fit) regression.

    Explanation of Regression Orders:
      - **Order 0**: Equivalent to the Nadaraya-Watson estimator. It calculates a weighted average where weights
        are determined by the Gaussian kernel and the validity (non-NaN status) of neighboring pixels.
      - **Order 1 & 2**: Local Weighted Least Squares (LOESS). It fits a polynomial surface (plane or quadratic) to
        the local neighborhood weighted by the Gaussian kernel. This acts as a robust 2D Savitzky-Golay filter.

    Mathematical basis:
      - Approximate a signal s(x, y) from noisy data f(x, y) = s(x, y) + e(x, y) using weighted local regression.
      - The approximation b(x, y) is calculated as the fitted value at point (x, y) using a weighted least squares
        approach. Weights are non-zero within the neighborhood [x - rx, x + rx] and [y - ry, y + ry], following a
        Gaussian distribution with standard deviations proportional to rx and ry.
      - Optimization:
        For **Order 0**, the operation is mathematically equivalent to a normalized convolution. This implementation
        uses FFT-based convolution for performance gains compared to pixel-wise regression.

    :param data: 2D input array containing float data. May contain NaNs.
    :param cutoff_pixels: The filter cutoff wavelength in pixels as array [cutoff_y, cutoff_x].
    :param regression_order: RegressionOrder enum specifying the polynomial fit order:
        GAUSSIAN_WEIGHTED_AVERAGE (0) = Gaussian weighted average.
        LOCAL_PLANAR (1) = Local planar fit (corrects for tilt).
        LOCAL_QUADRATIC (2) = Local quadratic fit (corrects for quadratic curvature).
    :returns: The filtered 2D array of the same shape as input.
    """
    alpha = (
        ALPHA_REGRESSION
        if regression_order == RegressionOrder.LOCAL_QUADRATIC
        else ALPHA_GAUSSIAN
    )
    kernel_x, kernel_y = _create_normalized_separable_kernels(alpha, cutoff_pixels)

    if regression_order == RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE:
        return _apply_order0_filter(data, kernel_x, kernel_y)

    return calculate_polynomial_filter(
        data,
        kernel_x,
        kernel_y,
        exponents=generate_polynomial_exponents(regression_order.value),
    )
