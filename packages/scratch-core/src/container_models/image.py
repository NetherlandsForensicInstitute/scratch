import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.constants import femto

from container_models.base import (
    BinaryMask,
    Coordinate,
    DepthData,
    FloatArray1D,
    Scale,
    ImageRGBA,
)


class MetaData(BaseModel):
    scale: Scale

    @property
    def is_isotropic(self) -> bool:
        return bool(np.isclose(self.scale.x, self.scale.y, atol=femto))

    @property
    def central_diff_scales(self) -> Scale:
        return self.scale.map(lambda x: x / 2)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        regex_engine="rust-regex",
    )


class ImageContainer(BaseModel):
    data: DepthData
    metadata: MetaData

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        regex_engine="rust-regex",
    )

    @property
    def height(self) -> int:
        """Return the height (number of rows) of the image."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Return the width (number of columns) of the image."""
        return self.data.shape[1]

    @property
    def rgba(self) -> ImageRGBA:
        gray_uint8 = np.nan_to_num(self.data, nan=0.0).astype(np.uint8)
        rgba = np.repeat(gray_uint8[..., np.newaxis], 4, axis=-1)
        rgba[..., 3] = (~np.isnan(self.data)).astype(np.uint8) * 255
        return rgba

    @property
    def valid_mask(self) -> BinaryMask:
        """Mask of the valid pixels in the data."""
        valid_mask = ~np.isnan(self.data)
        valid_mask.setflags(write=False)
        return valid_mask

    @property
    def valid_data(self) -> FloatArray1D:
        """Valid pixels in the data."""
        valid_data = self.data[self.valid_mask]
        valid_data.setflags(write=False)
        return valid_data

    @property
    def center(self) -> Coordinate:
        return self.metadata.central_diff_scales.map(
            lambda x, y: (x - 1) * y, other=reversed(self.data.shape)
        )
