from abc import ABC, abstractmethod
from typing import cast
from container_models.scan_image import ScanImage
from container_models.base import FloatArray2D, BinaryMask
from conversion.filter import ALPHA_GAUSSIAN, ALPHA_REGRESSION
from conversion.filter.gaussian import (
    create_normalized_separable_kernels,
    apply_order0_filter,
)
from conversion.leveling.data_types import SurfaceTerms
import numpy as np
from loguru import logger
from renders.computation import (
    DesignMatrixSolver,
    calculate_polynomial_filter,
    fit_surface,
    generate_point_cloud,
    generate_polynomial_exponents,
)
from skimage.transform import resize
from container_models.base import Point
from renders.computation import solve_least_squares
from utils.constants import RegressionOrder


class ImageMutation(ABC):
    """Represents a mutation on a ScanImage"""

    @abstractmethod
    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        raise NotImplementedError("jammer man")

    def mask_bounding_box(
        self, shape: FloatArray2D | BinaryMask
    ) -> tuple[slice, slice]:
        coordinates = np.nonzero(shape)
        y_min, x_min = np.min(coordinates, axis=1)
        y_max, x_max = np.max(coordinates, axis=1)
        return slice(x_min, x_max + 1), slice(y_min, y_max + 1)


class Mask(ImageMutation):
    def __init__(self, mask: BinaryMask) -> None:
        self.mask = mask

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """Apply the mask to the image data by setting masked-out pixels to NaN."""
        logger.info("Applying mask to scan_image")
        scan_image.data[~self.mask] = np.nan
        return scan_image


class Crop(ImageMutation):
    def __init__(self, crop: BinaryMask) -> None:
        self.crop = crop

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Crop the image to the bounding box of the mask.
        :returns: New ScanImage cropped to the minimal bounding box containing all True mask values.
        :raises ValueError: If the image does not contain a mask.
        """
        y_slice, x_slice = self.mask_bounding_box(self.crop)
        scan_image.data = scan_image.data[y_slice, x_slice]
        self.crop = self.crop[y_slice, x_slice]
        return scan_image


class Resample(ImageMutation):
    def __init__(self, factors: Point[float]) -> None:
        self.factors = factors

    def _resample_image_array[T: BinaryMask | FloatArray2D](
        self, array: T, factors: Point[float]
    ) -> T:
        """
        Resample an array using the specified resampling factors.
        For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.
        :param array: The array containing the image data to resample.
        :param factors: The multipliers for the scale of the X- and Y-axis.
        :returns: A numpy array containing the resampled image data.
        """
        output_shape = (1 / factors.y * array.shape[0], 1 / factors.x * array.shape[1])
        resampled = resize(
            image=array,
            output_shape=output_shape,
            mode="edge",
            anti_aliasing=array.dtype != np.bool_
            and all(factor > 1 for factor in factors),
        )
        logger.debug(
            f"Resampling image array to new size: {output_shape[0]}/{output_shape[1]}"
        )
        return cast(T, np.asarray(resampled, dtype=array.dtype))

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Resample the ScanImage object using the specified resampling factors.
        :param image: Input ScanImage to resample.
        :param factors: The multipliers for the scale of the X- and Y-axis.
        :returns: The resampled ScanImage.
        """
        return ScanImage(
            data=self._resample_image_array(scan_image.data, factors=self.factors),
            scale_x=scan_image.scale_x * self.factors.x,
            scale_y=scan_image.scale_y * self.factors.y,
        )


class LevelMap(ImageMutation):
    def __init__(
        self,
        terms: SurfaceTerms,
        solver: DesignMatrixSolver,
        reference_point: Point[float],
    ) -> None:
        self.terms = terms
        self.solver = solver
        self.reference_point = reference_point

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Compute the leveled map by fitting polynomial terms and subtracting them from the image data.
        This computation effectively acts as a high-pass filter on the image data.
        :param scan_image: The scan image containing the image data to level.
        :param terms: The surface terms to use in the polynomial fitting. Can be combined using bit-operators.
        :param reference_point: Tuple (X, Y) in physical coordinate space defining the origin for coordinate translation.
            The coordinates will be translated such that (X, Y) lies at the origin.
        :returns: 2D array containing the leveled scan data (original data minus fitted surface).
        """
        point_cloud = generate_point_cloud(
            scan_image=scan_image, reference_point=self.reference_point
        )
        fitted_surface = fit_surface(
            point_cloud=point_cloud, terms=self.terms, solver=self.solver
        )
        leveled_map_2d = np.full_like(scan_image.data, np.nan)
        leveled_map_2d[scan_image.valid_mask] = point_cloud.zs - fitted_surface

        scan_image.data = leveled_map_2d
        return scan_image


class GausionFilter(
    ImageMutation
):  # probably you want here a class FilterImage... with a filter as parameter. so you can choose what kind of filter
    def __init__(
        self,
        cutoff_pixels: FloatArray2D,
        regression_order: RegressionOrder,
    ) -> None:
        self.cutoff_pixels = cutoff_pixels
        self.regression_order = regression_order

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
            ALPHA_REGRESSION
            if self.regression_order == RegressionOrder.LOCAL_QUADRATIC
            else ALPHA_GAUSSIAN
        )
        kernel_x, kernel_y = create_normalized_separable_kernels(
            alpha, self.cutoff_pixels
        )

        if self.regression_order == RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE:
            return apply_order0_filter(scan_image.data, kernel_x, kernel_y)  # type: ignore

        scan_image.data = calculate_polynomial_filter(
            scan_image.data,
            kernel_x,
            kernel_y,
            exponents=generate_polynomial_exponents(self.regression_order.value),
        )
        return scan_image


def calculate_image(scan_image: ScanImage, mutations: list[ImageMutation]) -> ScanImage:
    for mutation in mutations:
        scan_image = mutation.apply_on_image(scan_image)
    return scan_image


if __name__ == "__main__":
    # fake edit endpoint
    # fills in all variables from received values from java
    scan_image = ScanImage(data=np.ones((5, 5), dtype=np.float64), scale_x=1, scale_y=1)
    # creates pipeline  with neccesarry values (mostly filled in from endpoint input, some calculated above if needed)
    pipeline: list[ImageMutation] = [
        Resample(factors=Point(2, 2)),
        Mask(mask=np.zeros((5, 5), dtype=bool)),
        Crop(crop=np.zeros((5, 5), dtype=bool)),
        LevelMap(
            terms=SurfaceTerms.ASTIG_0,
            solver=solve_least_squares,
            reference_point=Point(3, 3),
        ),
        GausionFilter(
            cutoff_pixels=np.ones((5, 5)),
            regression_order=RegressionOrder.GAUSSIAN_WEIGHTED_AVERAGE,
        ),
    ]
    # calculates the new image
    edited_scan_image = calculate_image(scan_image=scan_image, mutations=pipeline)
    # Tada done.
