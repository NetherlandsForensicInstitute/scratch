from functools import cached_property
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from .base import ScanMap2DArray, ConfigBaseModel, MaskArray


class ScanImage(ConfigBaseModel):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: ScanMap2DArray
    mask: ScanMap2DArray | None = None
    scale_x: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    meta_data: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mask_and_data_shape_matches(self) -> Self:
        if self.mask is not None and self.data.shape != self.mask.shape:
            raise ValueError(
                f"The shape of the data {self.data.shape}"
                f" does not match the shape of the mask {self.mask.shape}."
            )
        return self

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    @cached_property
    def valid_mask(self) -> MaskArray:
        """Mask of the valid pixels in the data."""
        valid_mask = ~np.isnan(self.data)
        valid_mask.setflags(write=False)
        return valid_mask

    @cached_property
    def valid_data(self) -> NDArray[np.floating]:
        """Valid pixels in the data."""
        valid_data = self.data[self.valid_mask]
        valid_data.setflags(write=False)
        return valid_data

    def model_copy(self, *, update=None, deep=False):
        copy = super().model_copy(update=update, deep=deep)
        # Invalidate cached properties when any field changes
        if update:
            # Dynamically find and clear all cached_property attributes
            for name in dir(type(copy)):
                attr = getattr(type(copy), name, None)
                if isinstance(attr, cached_property):
                    copy.__dict__.pop(name, None)
        return copy
