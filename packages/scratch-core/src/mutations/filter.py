import numpy as np
from loguru import logger
from surfalize import Surface
from container_models.base import BinaryMask
from container_models.scan_image import ScanImage
from conversion.filter import apply_gaussian_regression_filter
from conversion.leveling.data_types import SurfaceTerms
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation
from utils.constants import RegressionOrder


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
            # We need at least 3 terms for the least squares solver
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


class GausianRegressionFilter(ImageMutation):  # pragma: no cover
    NAN_OUT = True
    IS_HIGHT_PASS = False

    def __init__(self, cutoff_length: float, regression_order: RegressionOrder) -> None:
        self.cutoff_length = cutoff_length
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
        :param scan_image: Gausian filter is applied on this scan_image data.
        :returns: ScanImage with the filtered 2D array.
        """

        pixel_size = (scan_image.scale_y, scan_image.scale_x)
        scan_image.data = apply_gaussian_regression_filter(
            data=scan_image.data,
            cutoff_length=self.cutoff_length,
            pixel_size=pixel_size,
            regression_order=self.regression_order.value,
            nan_out=self.NAN_OUT,
            is_high_pass=self.IS_HIGHT_PASS,
        )
        return scan_image
