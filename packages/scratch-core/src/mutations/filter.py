from typing import NamedTuple, cast

import numpy as np
from loguru import logger

from container_models.base import BinaryMask, Coordinate, FloatArray1D, FloatArray2D
from container_models.image import ImageContainer
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.transforms import normalize_coordinates
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation
from renders.grid import get_2d_grid
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

    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Apply the mask to the image.

        :params scan_image: Input scan image to which the mask is applied.
        :return: The masked scan image.
        :raises ImageShapeMismatchError: If the mask shape does not match the image data shape.
        """
        if self.mask.shape != image.data.shape:
            raise ImageShapeMismatchError(
                f"Mask shape: {self.mask.shape} does not match image shape: {image.data.shape}"
            )
        logger.info("Applying mask to scan_image")
        image.data[~self.mask] = np.nan
        return image


class LevelMap(ImageMutation):
    """
    Image mutation that performs surface leveling by fitting and subtracting
    a polynomial surface from a scan image.

    The valid pixels of the input `ImageContainer` are interpreted as a 3D point
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

    def __init__(self, reference: Coordinate, terms: SurfaceTerms) -> None:
        self.reference = reference
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
        return cast(FloatArray1D, coefficients)

    def _evaluate_fitted_surface(
        self, point_cloud: PointCloud, terms: SurfaceTerms
    ) -> FloatArray1D:
        """
        Core solver: fits a surface to the point cloud.
        :param point_cloud: PointCloud containing the X, Y, and Z coordinates.
        :param terms: The surface terms to use in the polynomial fitting.
        :returns: 1D array containing the fitted surface values (zs).
        """
        normalized = normalize_coordinates(point_cloud.xs, point_cloud.ys)  # type: ignore[arg-type]
        design_matrix = build_design_matrix(normalized.xs, normalized.ys, terms)
        coefficients = self.solve_least_squares(
            design_matrix=design_matrix, zs=point_cloud.zs
        )
        return design_matrix @ coefficients

    @staticmethod
    def _generate_point_cloud(
        image: ImageContainer, reference: Coordinate
    ) -> PointCloud:
        """
        Generate a 3D point cloud from an image with coordinates centered at a reference point.
        :param image: The image containing the height data and valid mask.
        :param reference: Physical coordinate to use as the origin.
        :returns: PointCloud containing the valid X, Y, and Z coordinates.
        """
        x_grid, y_grid = get_2d_grid(image, offset=reference)
        return PointCloud(
            xs=cast(FloatArray1D, x_grid[image.valid_mask]),
            ys=cast(FloatArray1D, y_grid[image.valid_mask]),
            zs=cast(FloatArray1D, image.data[image.valid_mask]),
        )

    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Compute the leveled map by fitting polynomial terms and subtracting them from the image data.
        This computation effectively acts as a high-pass filter on the image data.
        :param image: The image containing the depth data to level.
        :returns: Image with leveled data (original data minus fitted surface).
        """
        point_cloud = self._generate_point_cloud(image, self.reference)
        fitted_surface = self._evaluate_fitted_surface(
            point_cloud=point_cloud,
            terms=self.terms,
        )
        leveled_map_2d = np.full_like(image.data, np.nan)
        leveled_map_2d[image.valid_mask] = point_cloud.zs - fitted_surface

        image.data = leveled_map_2d
        return image


class GausianRegressionFilter(ImageMutation):
    NAN_OUT = True
    IS_HIGHT_PASS = False

    def __init__(self, cutoff_length: float, regression_order: RegressionOrder) -> None:
        self.cutoff_length = cutoff_length
        self.regression_order = regression_order

    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
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
        :param scan_image: Gausian filter is applied on this scan_image data.
        :returns: ScanImage with the filtered 2D array.
        """

        image.data = apply_gaussian_regression_filter(
            data=image.data,
            cutoff_length=self.cutoff_length,
            pixel_size=image.metadata.scale,
            regression_order=self.regression_order.value,
            nan_out=self.NAN_OUT,
            is_high_pass=self.IS_HIGHT_PASS,
        )
        return image
