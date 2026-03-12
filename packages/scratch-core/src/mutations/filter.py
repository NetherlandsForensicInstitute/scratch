from typing import NamedTuple

import numpy as np
from loguru import logger
from scipy.ndimage import generic_filter

from container_models.base import BinaryMask, FloatArray1D, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling.data_types import SurfaceTerms
from conversion.leveling.solver.design import build_design_matrix
from conversion.leveling.solver.grid import get_2d_grid
from conversion.leveling.solver.transforms import normalize_coordinates
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation
from mutations.spatial import Resample
from utils.constants import RegressionOrder


class PointCloud(NamedTuple):
    xs: FloatArray1D
    ys: FloatArray1D
    zs: FloatArray1D


class FilterMedian(ImageMutation):
    def __init__(self, filter_size: int):
        self.filter_size = filter_size

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Apply the median filter to the image.

        :params scan_image: Input scan image to which the mask is applied.
        :return: The filtered scan image.
        """
        if self.filter_size % 2 == 0:
            self.filter_size += 1

        filtered_image = generic_filter(
            scan_image.data,
            np.nanmedian,
            size=self.filter_size,
            mode="constant",
            cval=np.nan,
        ).astype(np.float64)

        return ScanImage(
            data=filtered_image,
            scale_x=scan_image.scale_x,
            scale_y=scan_image.scale_y,
        )


class FilterNeedles(ImageMutation):
    TARGET_SCALE = 7e-5
    MEDIAN_FILTER_SIZE = 5
    SMALL_STRIP_THRESHOLD = 20
    MEDIAN_FACTOR_CORRECTION_FACTOR = 6
    MEDIAN_FACTOR = 15.0

    def _calculate_subsampling_factor(self, scan_image_scale: float) -> int:
        """Calculate subsampling factor for computational efficiency.

        calculation is done using the TARGET_SCALE and the desired MEDIAN_FILTER_SIZE of the class variable
        :param scan_image_scale: The scale of the scan image of the calculation
        :return: int of the subsampling factor
        """
        return int(
            np.ceil(self.TARGET_SCALE / self.MEDIAN_FILTER_SIZE / scan_image_scale)
        )

    def _image_is_small_strip(self, image_width: int, image_height: int) -> bool:
        """Checks if the given image is a small strip.

        :param image_width: the width of the given scan_image
        :param image_height: the height of the given scan_image
        :return: bool True if the image is a small strip
        """
        return (
            image_width <= self.SMALL_STRIP_THRESHOLD
            or image_height <= self.SMALL_STRIP_THRESHOLD
        )

    def _get_residual_image(self, scan_image: ScanImage) -> FloatArray2D:
        """
        Apply median filtering to smooth the image and compute residuals as the difference between the input scan image and
        the median filtered image.

        If the image is large, it is downsampled before filtering and upsampled afterwards.
        If the image is a small strip of data (width or height <= SMALL_STRIP_THRESHOLD), the filter size is reduced to
        avoid too extensive smoothing.

        :param scan_image: Scan image to calculate residual image for.
        :return: Array of differences between the input scan_image.data and median filter smoothed version of that image.
        """
        # Check if the image is a small strip of data
        original_data = scan_image
        is_small_strip = self._image_is_small_strip(
            image_width=scan_image.width, image_height=scan_image.height
        )
        subsample_factor = self._calculate_subsampling_factor(scan_image.scale_x)
        filter_size = self.MEDIAN_FILTER_SIZE
        if is_small_strip:
            filter_size = int(np.round(np.sqrt(self.MEDIAN_FILTER_SIZE)))
            logger.debug(
                f"scan image is a small strip of data, updated filter_size to :{filter_size}"
            )
        median_filter = FilterMedian(filter_size=filter_size)
        filter_median_pipeline: list[ImageMutation] = [median_filter]
        if subsample_factor > 1 and not is_small_strip:
            logger.debug(
                "Large image detected, add downsampling before and upsampling after filter to reduce computation."
            )
            downsampler = Resample(
                target_shape=(
                    int(scan_image.height / subsample_factor),
                    int(scan_image.width / subsample_factor),
                )
            )
            upsampler = Resample(
                target_shape=(
                    scan_image.height,
                    scan_image.width,
                )
            )
            filter_median_pipeline: list[ImageMutation] = [
                downsampler,
                median_filter,
                upsampler,
            ]

        for mutation in filter_median_pipeline:
            scan_image = mutation(scan_image).unwrap()
        # Use slicing since the shape may deviate slightly after down- and upsampling
        residual_image = (
            original_data.data
            - scan_image.data[: original_data.height, : original_data.width]
        )
        return residual_image

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Filter needles of the image.

        :params scan_image: Input scan image to which the mask is applied.
        :return: The filtered scan image.
        """
        median_factor = self.MEDIAN_FACTOR * self.MEDIAN_FACTOR_CORRECTION_FACTOR
        logger.debug(f"median factor: {median_factor}")
        residual_image = self._get_residual_image(scan_image)
        # Find needles: points where |residual| > threshold
        median_residual = np.nanmedian(np.abs(residual_image))
        threshold = median_factor * median_residual
        needles_mask = np.abs(residual_image) > threshold

        logger.info("Removing needles from scan image with a mask.")
        return Mask(mask=~needles_mask)(scan_image).unwrap()


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


class GaussianRegressionFilter(ImageMutation):  # pragma: no cover
    NAN_OUT = True

    def __init__(
        self,
        cutoff_length: float,
        regression_order: RegressionOrder,
        is_high_pass: bool,
    ) -> None:
        """
        :param cutoff_length: Filter cutoff wavelength in meters (m).
        :param regression_order: Order of the local polynomial fit.
        :param is_high_pass: Whether to use a high-pass filter.
        """
        self.cutoff_length = cutoff_length
        self.regression_order = regression_order
        self.is_high_pass = is_high_pass

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

        :param scan_image: Gaussian filter is applied on this scan_image data.
        :returns: ScanImage with the filtered 2D array.
        """
        pixel_size = (scan_image.scale_y, scan_image.scale_x)
        scan_image.data = apply_gaussian_regression_filter(
            data=scan_image.data,
            cutoff_length=self.cutoff_length,
            pixel_size=pixel_size,
            regression_order=self.regression_order.value,
            nan_out=self.NAN_OUT,
            is_high_pass=self.is_high_pass,
        )
        return scan_image
