"""Shading image mutations.

This module contains mutations that apply lighting and shading effects.

.. seealso::

    :class:`LightIntensityMap`
        Compute surface shading from directional light sources.
"""

from typing import Iterable, override

from returns.result import safe

from container_models.base import UnitVector
from container_models.image import ImageContainer
from mutations.base import ImageMutation
from renders.normalizations import (
    normalize_2d_array,
    normalize_to_surface_normals,
)
from renders.shading import combine_light_components
import numpy as np

from utils.logger import log_railway_function


class LightIntensityMap(ImageMutation):
    def __init__(
        self, light_sources: Iterable[UnitVector], observer: UnitVector
    ) -> None:
        self.light_sources = light_sources
        self.observer = observer

    @log_railway_function(
        "Failed to apply lights",
        "Successfully applied lights",
    )
    @safe
    def __call__(self, image: ImageContainer) -> ImageContainer:
        return self.apply_on_image(image)

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Combine multiple directional light sources into a single intensity map.
        """
        normalize_data = normalize_to_surface_normals(
            image.data, image.metadata.central_diff_scales
        )
        data = np.nansum(
            np.stack(
                [
                    combine_light_components(normalize_data, light, self.observer)
                    for light in self.light_sources
                ],
                axis=-1,
            ),
            axis=2,
        )
        image.data = normalize_2d_array(data)
        return image


# TODO: Needs better name
class ImageForDisplay(ImageMutation):
    def __init__(self, std_scaler: float) -> None:
        if std_scaler <= 0.0:
            raise ValueError("`std_scaler` must be a positive number.")
        self.std_scaler = std_scaler

    @log_railway_function(
        "Failed to retrieve array for display",
        "Successfully retrieve array for display",
    )
    @safe
    def __call__(self, image: ImageContainer) -> ImageContainer:
        return self.apply_on_image(image)

    @override
    def apply_on_image(self, image: ImageContainer) -> ImageContainer:
        """
        Clip and normalize image data for displaying purposes.

        First the data will be clipped to the interval [μ - σ * S, μ + σ * S].
        Then the values are min-max normalized and scaled to the [0, 255] interval.
        """
        mean = np.nanmean(image.data)
        std = np.nanstd(image.data, ddof=1) * self.std_scaler
        lower, upper = mean - std, mean + std
        clipped_data = np.clip(image.data, lower, upper)
        image.data = (clipped_data - lower) / (upper - lower) * 255
        return image
