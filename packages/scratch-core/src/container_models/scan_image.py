from functools import cached_property

import numpy as np
from pydantic import Field
from .base import ConfigBaseModel, BinaryMask, FloatArray1D, DepthData
from .base import ConfigBaseModel, BinaryMask, FloatArray1D, DepthData


class ScanImage(ConfigBaseModel):
    """
    A 2D image/array of floats.

    Used for: depth maps, intensity maps, single-channel images.
    Shape: (height, width)
    """

    data: DepthData
    scale_x: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    scale_y: float = Field(..., gt=0.0, description="pixel size in meters (m)")
    meta_data: dict = Field(default_factory=dict)

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]

    @cached_property
    def valid_mask(self) -> BinaryMask:
        """Mask of the valid pixels in the data."""
        valid_mask = ~np.isnan(self.data)
        valid_mask.setflags(write=False)
        return valid_mask

    @cached_property
    def valid_data(self) -> FloatArray1D:
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
