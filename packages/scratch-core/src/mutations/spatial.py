from typing import cast
from container_models.base import BinaryMask, FloatArray2D, Factors
from container_models.scan_image import ScanImage
from loguru import logger
from mutations.base import ImageMutation
from skimage.transform import resize
import numpy as np


class Resample(ImageMutation):
    def __init__(self, factors: Factors[float]) -> None:
        """
        Constructor for initiating the Resampling.

        :param factors: The multipliers for the scale of the X- and Y-axis.
        """
        self.factors = factors

    def _resample_image_array[T: BinaryMask | FloatArray2D](self, array: T) -> T:
        """
        Resample an array using the specified resampling factors.
        For example, if the scale factor is 0.5, then the image output shape will be scaled by 1 / 0.5 = 2.

        :param array: The array containing the image data to resample.
        :returns: A numpy array containing the resampled image data.
        """
        output_shape = (
            1 / self.factors.y * array.shape[0],
            1 / self.factors.x * array.shape[1],
        )
        resampled = resize(
            image=array,
            output_shape=output_shape,
            mode="edge",
            anti_aliasing=array.dtype != np.bool_
            and all(factor > 1 for factor in self.factors),
        )
        logger.debug(
            f"Resampling image array to new size: {output_shape[0]}/{output_shape[1]}"
        )
        return cast(T, np.asarray(resampled, dtype=array.dtype))

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Resample the ScanImage object using the specified resampling factors.

        :param image: Input ScanImage to resample.
        :returns: The resampled ScanImage.
        """
        return ScanImage(
            data=self._resample_image_array(scan_image.data),
            scale_x=scan_image.scale_x * self.factors.x,
            scale_y=scan_image.scale_y * self.factors.y,
        )
