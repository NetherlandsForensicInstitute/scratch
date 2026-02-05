"""
Spatial Image Mutations
=======================

This module contains image mutations that operate on the *spatial*
properties of a `ImageContainer`.

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

from typing import cast, override

import numpy as np
from loguru import logger
from skimage.transform import resize

from computations.spatial import get_bounding_box
from container_models.base import BinaryMask, DepthData, Factor, Pair
from container_models.image import ImageContainer, MetaData
from exceptions import ImageShapeMismatchError
from mutations.base import ImageMutation


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

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Crop the image to the bounding box of the mask.
        :returns: New ImageContainer cropped to the minimal bounding box containing all True mask values.
        """
        if image.data.shape != self.mask.shape:
            raise ImageShapeMismatchError(
                f"image shape: {image.data.shape} and crop shape: {self.mask.shape} are not equal"
            )
        y_slice, x_slice = get_bounding_box(self.mask)
        image.data = image.data[y_slice, x_slice]
        return image


class Resample(ImageMutation):
    def __init__(self, factors: Factor) -> None:
        """
        Constructor for initiating the Resampling.

        :param factors: The multipliers for the scale of the X- and Y-axis.
        """
        self.factors = factors

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Resample the ImageContainer object using the specified resampling factors.

        :param image: Input ImageContainer to resample.
        :returns: The resampled ImageContainer.
        """
        output_shape = Pair(*reversed(image.data.shape)) / self.factors
        resampled_data = resize(
            image=image.data,
            output_shape=output_shape,
            mode="edge",
            anti_aliasing=all(np.greater(self.factors, 1)),
        )
        logger.debug(
            f"Resampling image array to new size: {round(output_shape[0], 1)}/{round(output_shape[1], 1)}"
        )
        return ImageContainer(
            data=cast(DepthData, resampled_data).astype(image.data.dtype),
            metadata=MetaData(scale=image.metadata.scale * self.factors),
        )
