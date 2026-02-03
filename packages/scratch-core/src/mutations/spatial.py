"""
Spatial Image Mutations
=======================

This module contains image mutations that operate on the *spatial*
properties of a `ScanImage`.

Spatial mutations:
- change the geometric arrangement of pixels
- affect spatial resolution, scale, or coordinate systems

These mutations modify *where* pixels are located or how they are
interpreted in space, without changing the meaning or intensity of
the pixel data itself.

Typical examples include:
- resampling or resizing
- cropping to a region of interest
- scaling or coordinate transforms

All mutations in this module must preserve the semantic content of
the image while adjusting its spatial representation.
"""

from container_models.base import BinaryMask, DepthData
from computations.spatial import get_bounding_box
from container_models.scan_image import ScanImage
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation

import numpy as np
from loguru import logger
from pydantic import PositiveFloat
from skimage.transform import resize
from typing import cast


class CropToMask(ImageMutation):
    def __init__(self, mask: BinaryMask) -> None:
        if not np.any(mask):
            raise ValueError("Can't crop to a mask where there are only 0/False")
        self.mask = mask

    @property
    def skip_predicate(self) -> bool:
        """
        Determine whether this crop should be skipped.

        Skips computation if the crop contains only ones.

        :returns: True if the crop is empty, False otherwise
        """
        return bool(self.mask.all())

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Crop the image to the bounding box of the mask.
        :returns: New ScanImage cropped to the minimal bounding box containing all True mask values.
        """
        if scan_image.data.shape != self.mask.shape:
            raise ImageShapeMismatchError(
                f"image shape: {scan_image.data.shape} and crop shape: {self.mask.shape} are not equal"
            )
        y_slice, x_slice = get_bounding_box(self.mask)
        scan_image.data = scan_image.data[y_slice, x_slice]
        return scan_image


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
            anti_aliasing=self.x_factor > 1 and self.y_factor > 1,
        )
        logger.debug(
            f"Resampling image array to new size: {round(output_shape[0], 1)}/{round(output_shape[1], 1)}"
        )
        return ScanImage(
            data=cast(DepthData, resampled_data).astype(scan_image.data.dtype),
            scale_x=scan_image.scale_x * self.x_factor,
            scale_y=scan_image.scale_y * self.y_factor,
        )
