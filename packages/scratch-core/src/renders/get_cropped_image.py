from enum import Enum
from itertools import product
from typing import Protocol

from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.transforms import normalize_coordinates
import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from container_models.base import ScanMap2DArray, PointCloud
from container_models.scan_image import ScanImage
from conversion.filter import (
    _apply_order0_filter,
    _build_lhs_matrix,
    _convolve_2d_separable,
    _create_normalized_separable_kernels,
    _solve_pixelwise_regression,
)
from conversion.leveling import SurfaceTerms
from conversion.leveling.solver.grid import get_2d_grid
from conversion.leveling.solver.utils import (
    compute_image_center,
)
from conversion.resample import _clip_factors

# TODO: Based on what this code is doing. We can ignore this function completely
# [] First breakdown the code smells below, make sure they give the same response
# [x] extract resample_scan_image, resample_array (mask_and_crop_scan_image)
# [x] extract mask_scan_image, crop_scan_image (mask_and_crop_scan_image)
# [] extract level_map
# [] extroct apply_gaussian_regression_filter
# [] Remove get_cropped_image


# Constants based on ISO 16610 surface texture standards
# Standard Gaussian alpha for 50% transmission
ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
# Adjusted alpha often used for higher-order regression filters to maintain properties
# alpha = Sqrt((-1 - LambertW(-1, -1 / (2 * exp(1)))) / Pi)
ALPHA_REGRESSION = 0.7309134280946760


class RegressionOrder(Enum):
    GAUSSIAN_WEIGHTED_AVERAGE = 0
    LOCAL_PLANAR = 1
    LOCAL_QUADRATIC = 2


def _get_polynomial_exponents(order: int) -> list[tuple[int, int]]:
    return [
        (py, px) for py, px in product(range(order + 1), repeat=2) if py + px <= order
    ]


def _apply_polynomial_filter(
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
    :param order: The degree of the polynomial to fit (typically 1 for linear or 2 for quadratic).
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
    :param cutoff_length: The filter cutoff wavelength in physical units.
    :param pixel_size: Tuple of (y_size, x_size) in physical units.
    :param regression_order: Order of the local polynomial fit (0, 1, or 2).
        0 = Gaussian weighted average.
        1 = Local planar fit (corrects for tilt).
        2 = Local quadratic fit (corrects for quadratic curvature).
    :param nan_out: If True, input NaNs remain NaNs in output. If False, the filter attempts to
        fill gaps based on the local regression.
    :param is_high_pass: If True, returns (input - smoothed). If False, returns smoothed.
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

    exponents = _get_polynomial_exponents(regression_order.value)
    return _apply_polynomial_filter(data, kernel_x, kernel_y, exponents)


def resample_scan_image(image: ScanImage, factors: tuple[float, float]) -> ScanImage:
    """
    Resample the ScanImage object using the specified resampling factors.

    :param image: Input ScanImage to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: The resampled ScanImage.
    """
    mask = (
        None
        if image.mask is None
        else resample_image_array(image.mask, factors=factors)
    )
    return ScanImage(
        data=resample_image_array(image.data, factors=factors),
        scale_x=image.scale_x * factors[0],
        scale_y=image.scale_y * factors[1],
        mask=mask,
    )


def resample_image_array(
    array: NDArray,
    factors: tuple[float, float],
) -> NDArray:
    """
    Resample an array using the specified resampling factors.

    For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

    :param array: The array containing the image data to resample.
    :param factors: The multipliers for the scale of the X- and Y-axis.
    :returns: A numpy array containing the resampled image data.
    """
    factor_x, factor_y = factors
    resampled = resize(
        image=array,
        output_shape=(1 / factor_y * array.shape[0], 1 / factor_x * array.shape[1]),
        mode="edge",
        anti_aliasing=array.dtype != np.bool_ and all(factor > 1 for factor in factors),
    )
    return np.asarray(resampled, dtype=array.dtype)


def mask_image(image: ScanImage) -> ScanImage:
    """Apply the mask to the data."""
    image = image.model_copy()
    image.data[~image.mask] = np.nan  # type: ignore[index]
    return image


def _determine_bounding_box(mask: ScanMap2DArray) -> tuple[slice, slice]:
    """
    Determines the bounding box of non-zero values in a mask.

    :param mask: Binary mask array
    :return: Tuple of (y_slice, x_slice) for the bounding box
    """
    non_zero_coords = np.nonzero(mask)
    if not non_zero_coords[0].size:
        raise ValueError("Mask is empty")

    y_min, x_min = np.min(non_zero_coords, axis=1)
    y_max, x_max = np.max(non_zero_coords, axis=1)
    return slice(x_min, x_max + 1), slice(y_min, y_max + 1)


def crop_to_mask(image: ScanImage) -> ScanImage:
    """Crop to the bounding box of the mask."""
    if image.mask is None:
        raise ValueError("Mask is required for cropping operation.")
    y_slice, x_slice = _determine_bounding_box(image.mask)
    return image.model_copy(
        update={
            "data": image.data[y_slice, x_slice],
            "mask": None if image.mask is None else image.mask[y_slice, x_slice],
        }
    )


def get_fitted_surface_2d(
    scan_image: ScanImage, fitted_surface: NDArray[np.float64]
) -> ScanMap2DArray:
    """ "Compute fitted surface in 2d from a 3d"""
    fitted_surface_2d = np.full_like(scan_image.data, np.nan)
    fitted_surface_2d[scan_image.valid_mask] = fitted_surface
    return fitted_surface_2d


def generate_point_cloud(
    scan_image: ScanImage, reference_point: tuple[float, float]
) -> PointCloud:
    # Build the 2D grids and translate in the opposite direction of `reference_point`
    x_grid, y_grid = get_2d_grid(
        scan_image, offset=(-reference_point[0], -reference_point[1])
    )
    # Get the point cloud (xs, ys, zs) for the numerical data
    xs, ys, zs = (
        x_grid[scan_image.valid_mask],
        y_grid[scan_image.valid_mask],
        scan_image.valid_data,
    )
    return PointCloud(xs=xs, ys=ys, zs=zs)


class CoefficientsProtocol(Protocol):
    def __call__[T: np.number](
        self, design_matrix: NDArray[T], zs: NDArray
    ) -> NDArray[T]: ...


def fit_surface(
    point_cloud: PointCloud, terms: SurfaceTerms, solver: CoefficientsProtocol
) -> NDArray[np.float64]:
    """
    Core solver: fits a surface to the point cloud (xs, ys, zs).

    :param xs: The X-coordinates.
    :param ys: The Y-coordinates.
    :param zs: The Z-values.
    :param terms: The terms to use in the fitting
    :return: A tuple containing the fitted surface (z̃s) and the estimated physical parameters.
    """
    # 1. Normalize the grid coordinates by centering and rescaling them
    normalized = normalize_coordinates(point_cloud.xs, point_cloud.ys)

    # 2. Build the design matrix for the least-squares solver
    design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)

    # 3. Solve (Least Squares)
    coefficients = solver(design_matrix=design_matrix, zs=point_cloud.zs)
    # 4. Compute the surface (z̃s-values) from the fitted coefficients
    return design_matrix @ coefficients


def solve_least_squares(
    design_matrix: NDArray[np.float64], zs: NDArray
) -> NDArray[np.floating[any]]:
    (
        coefficients,
        *_,
    ) = np.linalg.lstsq(design_matrix, zs, rcond=None)
    return coefficients


def level_map(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    reference_point: tuple[float, float],
) -> ScanMap2DArray:
    """
    Compute the leveled map by fitting polynomial terms and subtracting them from the image data.

    This computation effectively acts as a high-pass filter on the image data.

    :param scan_image: The scan image containing the image data to level.
    :param terms: The surface terms to use in the fitting. Note: terms can be combined using bit-operators.
    :param reference_point: A tuple representing a reference point (X, Y) in physical coordinate space.
        If provided, then the coordinates will be translated such that (X, Y) lies in the origin after translation.
        If `None`, then the coordinates will be translated such that the center of the image lies in the origin.
    :returns: An instance of `LevelingResult` containing the leveled scan data and estimated physical parameters.
    """
    point_cloud = generate_point_cloud(
        scan_image=scan_image, reference_point=reference_point
    )
    fitted_surface = fit_surface(
        point_cloud=point_cloud, terms=terms, solver=solve_least_squares
    )
    leveled_map_2d = np.full_like(scan_image.data, np.nan)
    leveled_map_2d[scan_image.valid_mask] = point_cloud.zs - fitted_surface

    return leveled_map_2d


def get_cropped_image(
    scan_image: ScanImage,
    terms: SurfaceTerms,
    cutoff_length: float,
    regression_order: RegressionOrder,
    resampling_factors: tuple[float, float],
    crop: bool = False,
) -> NDArray:
    """
    Generate a preview image for the cropping editor by applying resampling, leveling, and filtering to depth data.

    :param scan_image: ScanImage to be processed.
    :param mask: Mask indicating fore/background to be applied to the data in `scan_image`.
    :param terms: The surface terms to be used in the fitting. Note: terms can be combined using bit-operators.
    :param cutoff_length: Cutoff wavelength in physical units.
    :param regression_order: Filter regression order used when filtering the data.
    :param resampling_factors: The resampling factors for the X- and Y-axis scales.
    :param crop: Whether to crop the result (i.e. remove outer NaNs).
    :returns: A numpy array with the cropped image data.
    """
    resampling_factors = _clip_factors(resampling_factors, True)
    scan_image = resample_scan_image(image=scan_image, factors=resampling_factors)
    if scan_image.mask is None:
        raise ValueError("Mask is required for cropping operation.")

    scan_image = mask_image(scan_image)
    if crop:
        scan_image = crop_to_mask(scan_image)

    center_x, center_y = compute_image_center(scan_image)
    level_result = level_map(
        scan_image=scan_image, terms=terms, reference_point=(center_x, center_y)
    )

    # Filter the leveled results using a Gaussian regression filter
    filtered_data = apply_gaussian_regression_filter(
        data=level_result,
        regression_order=regression_order,
        cutoff_pixels=cutoff_length
        / np.array([scan_image.scale_x, scan_image.scale_y]),
    )
    filtered_data[np.isnan(level_result)] = np.nan
    return level_result - filtered_data
