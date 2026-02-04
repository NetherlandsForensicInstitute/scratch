from collections import namedtuple
from itertools import product
from typing import NamedTuple

import numpy as np
from loguru import logger

from container_models.base import DepthData, FloatArray1D, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.filter.regression import (
    _build_lhs_matrix,
    _solve_pixelwise_regression,
    apply_order0_filter,
    convolve_2d_separable,
    create_normalized_separable_kernels,
)
from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.grid import get_2d_grid
from conversion.leveling.solver.transforms import normalize_coordinates
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation
from utils.constants import RegressionOrder


class PointCloud(NamedTuple):
    xs: FloatArray1D
    ys: FloatArray1D
    zs: FloatArray1D


class Mask(ImageMutation):
    """
    Image mutation that applies a binary mask to a scan image.

    All pixels corresponding to `False` (or zero) values in the mask
    are set to `np.nan` in the image data. Pixels where the mask is
    `True` remain unchanged.
    """

    def __init__(self, mask: BinaryMask) -> None:
        """
        Initialize the Mask mutation.

        :param mask: Binary mask indicating which pixels should be kept (`True`)
            or masked (`False`).
        """
        self.mask = mask

    @property
    def skip_predicate(self) -> bool:
        """
        Determine whether the masking operation can be skipped.

        If the mask contains no masked pixels (i.e. all values are `True`),
        applying the mask would have no effect and the mutation is skipped.

        :returns: bool `True` if the mutation can be skipped, otherwise `False`.
        """
        if self.mask.all():
            logger.warning(
                "skipping masking, Mask area is not containing any masking fields."
            )
            return True
        return False

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Apply the mask to the image.

        :params scan_image: Input scan image to which the mask is applied.
        :return: The masked scan image.
        :raises ImageShapeMismatchError: If the mask shape does not match the image data shape.
        """
        if self.mask.shape != scan_image.data.shape:
            raise ImageShapeMismatchError(
                f"Mask shape: {self.mask.shape} does not match image shape: {scan_image.data.shape}"
            )
        logger.info("Applying mask to scan_image")
        scan_image.data[~self.mask] = np.nan
        return scan_image


class LevelMap(ImageMutation):
    """
    Image mutation that performs surface leveling by fitting and subtracting
    a polynomial surface from a scan image.

    The valid pixels of the input `ScanImage` are interpreted as a 3D point
    cloud (X, Y, Z). A polynomial surface, defined by `SurfaceTerms`, is fitted
    to this data using a least-squares approach.
    The fitted surface is then subtracted from the original height data.

    Coordinates are translated such that the given reference point becomes
    the origin and are normalized for numerical stability.

    Parameters
    ----------
    x_reference_point : float
        X-coordinate used as the origin for surface fitting.
    y_reference_point : float
        Y-coordinate used as the origin for surface fitting.
    terms : SurfaceTerms
        Polynomial surface terms defining the fitted surface.
    """

    def __init__(
        self, x_reference_point: float, y_reference_point: float, terms: SurfaceTerms
    ) -> None:
        self.x_reference_point = x_reference_point
        self.y_reference_point = y_reference_point
        self.terms = terms

    @staticmethod
    def solve_least_squares(
        design_matrix: FloatArray2D, zs: FloatArray1D
    ) -> FloatArray1D:
        """
        Solve the least squares problem to find polynomial coefficients.
        :param design_matrix: The design matrix constructed from polynomial terms.
        :param zs: The Z-values (height data) to fit.
        :returns: Array of polynomial coefficients.
        """
        (
            coefficients,
            *_,
        ) = np.linalg.lstsq(design_matrix, zs, rcond=None)
        return coefficients

    def _evaluate_fitted_surface(
        self, point_cloud: PointCloud, terms: SurfaceTerms
    ) -> FloatArray1D:
        """
        Core solver: fits a surface to the point cloud.
        :param point_cloud: PointCloud containing the X, Y, and Z coordinates.
        :param terms: The surface terms to use in the polynomial fitting.
        :returns: 1D array containing the fitted surface values (zs).
        """
        normalized = normalize_coordinates(point_cloud.xs, point_cloud.ys)
        design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)
        coefficients = self.solve_least_squares(
            design_matrix=design_matrix, zs=point_cloud.zs
        )
        return design_matrix @ coefficients

    @staticmethod
    def _generate_point_cloud(
        scan_image: ScanImage, x_reference_point: float, y_reference_point: float
    ) -> PointCloud:
        """
        Generate a 3D point cloud from a scan image with coordinates centered at a reference point.
        :param scan_image: The scan image containing the height data and mask.
        :param x_reference_point: x in physical coordinates to use as the origin.
        :param y_reference_point: y in physical coordinates to use as the origin.
        :returns: PointCloud containing the valid X, Y, and Z coordinates.
        """
        x_grid, y_grid = get_2d_grid(
            scan_image, offset=(-x_reference_point, -y_reference_point)
        )
        xs, ys, zs = (
            x_grid[scan_image.valid_mask],
            y_grid[scan_image.valid_mask],
            scan_image.valid_data,
        )
        return PointCloud(xs=xs, ys=ys, zs=zs)

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Compute the leveled map by fitting polynomial terms and subtracting them from the image data.
        This computation effectively acts as a high-pass filter on the image data.
        :param scan_image: The scan image containing the image data to level.
        :returns: scan_image with the array containing the leveled scan data (original data minus fitted surface).
        """
        point_cloud = self._generate_point_cloud(
            scan_image=scan_image,
            y_reference_point=self.y_reference_point,
            x_reference_point=self.x_reference_point,
        )
        fitted_surface = self._evaluate_fitted_surface(
            point_cloud=point_cloud,
            terms=self.terms,
        )
        leveled_map_2d = np.full_like(scan_image.data, np.nan)
        leveled_map_2d[scan_image.valid_mask] = point_cloud.zs - fitted_surface

        scan_image.data = leveled_map_2d
        return scan_image


class GausianRegressionFilter(ImageMutation):
    # Constants based on ISO 16610 surface texture standards
    # Standard Gaussian alpha for 50% transmission
    ALPHA_GAUSSIAN = np.sqrt(np.log(2) / np.pi)
    # Adjusted alpha often used for higher-order regression filters to maintain properties
    # alpha = Sqrt((-1 - LambertW(-1, -1 / (2 * exp(1)))) / Pi)
    ALPHA_REGRESSION = 0.7309134280946760
    _Exponent = namedtuple("Exponent", ["y", "x"])

    def __init__(
        self, cutoff_pixels: FloatArray1D, regression_order: RegressionOrder
    ) -> None:
        self.cutoff_pixels = cutoff_pixels
        self.regression_order = regression_order

    def calculate_polynomial_filter(
        self,
        data: DepthData,
        kernel_x: FloatArray1D,
        kernel_y: FloatArray1D,
        exponents: list[_Exponent],
    ) -> DepthData:
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
                convolve_2d_separable(
                    weighted_data, (x_coords**px) * kernel_x, (y_coords**py) * kernel_y
                )
                for py, px in exponents
            ]
        )

        # Calculate LHS Matrix 'A' (Weight Moments)
        # A_jk = Convolution(weights, x^(px_j + px_k) * y^(py_j + py_k) * Kernel)
        lhs_matrix = _build_lhs_matrix(
            weights,
            kernel_x,
            kernel_y,
            x_coords,
            y_coords,
            exponents,  # type: ignore
        )

        # 3. Solve the System (A * c = b) per pixel
        return _solve_pixelwise_regression(lhs_matrix, rhs_moments, data)

    def generate_polynomial_exponents(self, order: int) -> list[_Exponent]:
        """
        Generate polynomial exponent pairs for 2D polynomial terms up to a given order.
        :param order: Maximum total degree (py + px) for the polynomial terms.
        :returns: List of (power_y, power_x) tuples representing polynomial terms.
        """
        return [
            self._Exponent(x, y)
            for y, x in product(range(order + 1), repeat=2)
            if y + x <= order
        ]

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
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
            self.ALPHA_REGRESSION
            if self.regression_order == RegressionOrder.LOCAL_QUADRATIC
            else self.ALPHA_GAUSSIAN
        )
        kernel_x, kernel_y = create_normalized_separable_kernels(
            alpha, self.cutoff_pixels
        )

        if self.regression_order == RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE:
            scan_image.data = apply_order0_filter(scan_image.data, kernel_x, kernel_y)
            return scan_image
        scan_image.data = self.calculate_polynomial_filter(
            scan_image.data,
            kernel_x,
            kernel_y,
            exponents=self.generate_polynomial_exponents(self.regression_order.value),
        )
        return scan_image
