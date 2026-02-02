from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.constants import femto

from container_models.base import DepthData


# TODO: Better Name
class Point[T](NamedTuple):
    x: T
    y: T


class MetaData(BaseModel):
    processing_history: list[str] = Field(default_factory=list)
    scale: Point[float]

    @property
    def is_isotropic(self) -> bool:
        return bool(np.isclose(self.scale.x, self.scale.y, atol=femto))

    @property
    def central_diff_scales(self) -> Point[float]:
        return Point(x=self.scale.x * 0.5, y=self.scale.y * 0.5)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        regex_engine="rust-regex",
    )


class ImageContainer(BaseModel):
    data: DepthData
    meta_data: MetaData

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        regex_engine="rust-regex",
    )

    @property
    def width(self) -> int:
        """The image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """The image height in pixels."""
        return self.data.shape[0]
