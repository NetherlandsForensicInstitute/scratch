from container_models.base import DepthData
from container_models.scan_image import ScanImage
from loguru import logger
from mutations.base import ImageMutation
from pydantic import PositiveFloat
from skimage.transform import resize
import numpy as np
from typing import cast


class Resample(ImageMutation):
    def __init__(self, x_factor: PositiveFloat, y_factor: PositiveFloat) -> None:
        """
        Constructor for initiating the Resampling.

        :param factors: The multipliers for the scale of the X- and Y-axis.
        """
        self.x_factor = x_factor
        self.y_factor = y_factor

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Resample the ScanImage object using the specified resampling factors.

        :param image: Input ScanImage to resample.
        :returns: The resampled ScanImage.
        """
        output_shape = (
            1 / self.y_factor * scan_image.height,
            1 / self.x_factor * scan_image.width,
        )
        resampled_data = resize(
            image=scan_image.data,
            output_shape=output_shape,
            mode="edge",
            anti_aliasing=scan_image.data.dtype != np.bool_
            and all(factor > 1 for factor in (self.x_factor, self.y_factor)),
        )
        logger.debug(
            f"Resampling image array to new size: {round(output_shape[0], 1)}/{round(output_shape[1], 1)}"
        )
        return ScanImage(
            data=cast(DepthData, resampled_data),
            scale_x=scan_image.scale_x * self.x_factor,
            scale_y=scan_image.scale_y * self.y_factor,
        )
