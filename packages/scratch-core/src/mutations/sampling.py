"""Sampling image mutations.

This module contains mutations that change image sampling or resolution.

.. seealso::

    :class:`IsotropicResample`
        Resample to equal pixel spacing in X and Y.
    :class:`Subsample`
        Reduce resolution by skipping pixels at regular intervals.
"""

from typing import override
from loguru import logger
from returns.result import safe
from container_models.base import DepthData, Pair
from container_models.image import ImageContainer
from mutations.base import ImageMutation
from utils.logger import log_railway_function
from numpy import float64, asarray
from skimage.transform import resize


class _IsotropicResample(ImageMutation):
    @log_railway_function(
        "Failed to make image resolution isotropic",
        "Successfully upsampled image file to isotropic resolution",
    )
    @safe
    def __call__(self, image: ImageContainer) -> ImageContainer:
        return image if image.metadata.is_isotropic else self.apply_on_image(image)

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Resample a scan image to isotropic resolution (i.e. equal pixel spacing in X and Y).

        Image is upsampled to the highest available resolution
        (the smaller of the two scale factors) using nearest-neighbor interpolation.
        Note: NaN values are preserved and will not be interpolated.
        """
        resolution = min(image.metadata.scale)
        shape = (
            round(image.height * image.metadata.scale.y / resolution),
            round(image.width * image.metadata.scale.x / resolution),
        )
        image.data = self._upsample_image(image.data, shape)
        image.metadata.scale = Pair(resolution, resolution)
        return image

    def _upsample_image(self, data: DepthData, shape: tuple[int, int]) -> DepthData:
        """Upsample image data to a common target scale using nearest-neighbor interpolation."""

        return asarray(
            resize(
                image=data,
                output_shape=shape,
                mode="edge",
                anti_aliasing=False,  # Disabled for pure upsampling
                preserve_range=True,  # Keep original data intensity levels
                order=0,  # Nearest Neighbor so that NaNs appear at corresponding coordinates
            ),
            dtype=float64,
        )


IsotropicResample = _IsotropicResample()


class Subsample(ImageMutation):
    def __init__(self, step_size_x: int, step_size_y: int) -> None:
        self.step_size_x = step_size_x
        self.step_size_y = step_size_y

    @property
    def skip_predicate(self) -> bool:
        return self.step_size_x == 1 and self.step_size_y == 1

    @log_railway_function(
        "Failed to subsample image file",
        "Successfully subsampled scan file",
    )
    @safe
    def __call__(self, image: ImageContainer) -> ImageContainer:
        if self.skip_predicate:
            logger.info("No subsampling needed, returning original scan image")
            return image
        if not (
            0 < self.step_size_x < image.width and 0 < self.step_size_y < image.height
        ):
            raise ValueError(
                f"Step size should be positive and smaller than the image size: ({image.height}, {image.width})"
            )
        return self.apply_on_image(image)

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """Subsample image by skipping steps in each dimension."""
        image.data = image.data[:: self.step_size_y, :: self.step_size_x].copy()
        image.metadata.scale = image.metadata.scale * Pair(
            self.step_size_x, self.step_size_y
        )
        return image
