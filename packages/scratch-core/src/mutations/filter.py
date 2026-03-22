import numpy as np
from loguru import logger
from surfalize import Surface
from scipy.ndimage import generic_filter

from computations.spatial import get_bounding_box
from container_models.base import BinaryMask, FloatArray2D
from container_models.scan_image import ScanImage
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling.data_types import SurfaceTerms
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation
from mutations.spatial import Resample
from utils.constants import RegressionOrder


class FilterMedian(ImageMutation):
    def __init__(self, filter_size: int):
        self.filter_size = filter_size

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Apply the median filter to the image.

        :params scan_image: Input scan image to filter
        :return: The filtered scan image.
        """
        filter_size = (
            self.filter_size if (self.filter_size % 2) else self.filter_size + 1
        )
        filtered_image = generic_filter(
            scan_image.data,
            np.nanmedian,
            size=filter_size,
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
    SMALL_STRIP_FILTER_SIZE = int(np.round(np.sqrt(MEDIAN_FILTER_SIZE)))
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
            np.ceil((self.TARGET_SCALE / self.MEDIAN_FILTER_SIZE) / scan_image_scale)
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
        original_scan_image = scan_image
        is_small_strip = self._image_is_small_strip(
            image_width=scan_image.width, image_height=scan_image.height
        )
        subsample_factor = self._calculate_subsampling_factor(scan_image.scale_x)
        filter_size = self.MEDIAN_FILTER_SIZE
        if is_small_strip:
            filter_size = self.SMALL_STRIP_FILTER_SIZE
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
            scan_image = mutation(scan_image)
        # Use slicing since the shape may deviate slightly after down- and upsampling
        return original_scan_image.data - scan_image.data

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
        return Mask(mask=~needles_mask)(scan_image)


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

    Parameters
    ----------
    terms : SurfaceTerms
        Polynomial surface terms defining the fitted surface.
    """

    def __init__(self, terms: SurfaceTerms) -> None:
        self.terms = terms

    @property
    def skip_predicate(self) -> bool:
        return self.terms == SurfaceTerms.NONE

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Compute the leveled map by fitting polynomial terms and subtracting them from the image data.
        This computation effectively acts as a high-pass filter on the image data.
        :param scan_image: The scan image containing the image data to level.
        :returns: scan_image with the array containing the leveled scan data (original data minus fitted surface).
        """
        if scan_image.valid_mask.sum() < 3:
            # We need at least 3 values for the least squares solver
            return scan_image

        surface = Surface(
            height_data=scan_image.data,
            step_x=scan_image.scale_x,
            step_y=scan_image.scale_y,
        )
        match self.terms:
            case (
                SurfaceTerms.TILT_X
                | SurfaceTerms.TILT_Y
                | SurfaceTerms.ASTIG_45
                | SurfaceTerms.PLANE
            ):
                degree = 1
            case SurfaceTerms.DEFOCUS | SurfaceTerms.ASTIG_0 | SurfaceTerms.SPHERE:
                degree = 2
            case _:
                degree = 0

        leveled, trend = surface.detrend_polynomial(
            degree=degree, inplace=False, return_trend=True
        )
        scan_image.data = leveled.data
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
        valid_region = get_bounding_box(mask=scan_image.valid_mask, margin=0)
        scan_image.data[valid_region] = apply_gaussian_regression_filter(
            data=scan_image.data[valid_region],
            cutoff_length=self.cutoff_length,
            pixel_size=pixel_size,
            regression_order=self.regression_order.value,
            nan_out=self.NAN_OUT,
            is_high_pass=self.is_high_pass,
        )
        return scan_image
