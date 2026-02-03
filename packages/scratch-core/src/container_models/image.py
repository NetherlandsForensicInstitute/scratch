from typing import NamedTuple

from PIL.Image import Image, fromarray
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.constants import femto

from container_models.base import ImageData, Pair, Scale


class Shape(NamedTuple):
    height: int
    width: int


class MetaData(BaseModel):
    processing_history: list[str] = Field(default_factory=list)
    scale: Scale

    @property
    def is_isotropic(self) -> bool:
        return bool(np.isclose(self.scale.x, self.scale.y, atol=femto))

    @property
    def central_diff_scales(self) -> Scale:
        return Pair[float](self.scale.x * 0.5, self.scale.y * 0.5)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        regex_engine="rust-regex",
    )


class ImageContainer(BaseModel):
    data: ImageData
    metadata: MetaData

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        regex_engine="rust-regex",
    )

    @property
    def shape(self) -> Shape:
        return Shape(*self.data.shape)

    @property
    def pil(self) -> Image:
        return fromarray(self.data)
