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

import numpy as np

from container_models.base import BinaryMask
from computations.spatial import get_bounding_box
from container_models.scan_image import ScanImage
from exceptions import ImageShapeMismatchError
from loguru import logger
from mutations.base import ImageMutation
from skimage.transform import resize


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
    def __init__(self, target_shape: tuple[float, float]) -> None:
        """
        Constructor for initiating the Resampling.

        :param target_shape: The multipliers for the scale of the X- and Y-axis(y first).
        """
        self.target_shape_height = target_shape[0]
        self.target_shape_width = target_shape[1]

    def apply_on_image(self, scan_image: ScanImage) -> ScanImage:
        """
        Resample the ScanImage object using the specified resampling factors.

        :param image: Input ScanImage to resample.
        :returns: The resampled ScanImage.
        """
        anti_aliasing = (
            self.target_shape_height < scan_image.height
            or self.target_shape_width < scan_image.width
        )
        resampled_data = resize(
            image=scan_image.data,
            output_shape=(
                self.target_shape_height,
                self.target_shape_width,
            ),
            mode="edge",
            anti_aliasing=anti_aliasing,
        )
        scale_x_factor = scan_image.width / self.target_shape_width
        scale_y_factor = scan_image.height / self.target_shape_height

        logger.debug(
            f"Resampling image array to new size: {round(self.target_shape_height, 1)}/{round(self.target_shape_width, 1)} with scale: x:{round(scale_x_factor, 1)}, y:{round(scale_y_factor, 1)}"
        )

        return ScanImage(
            data=np.asarray(resampled_data, dtype=scan_image.data.dtype),
            scale_x=scan_image.scale_x * scale_x_factor,
            scale_y=scan_image.scale_y * scale_y_factor,
        )
