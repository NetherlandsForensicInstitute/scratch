from typing import Annotated
from enum import StrEnum, auto
import json

from functools import partial
from pydantic import (
    Field,
    computed_field,
    AfterValidator,
    PlainSerializer,
    BeforeValidator,
)
from numpy import float64
from numpy.typing import NDArray
from container_models.base import (
    ConfigBaseModel,
    coerce_to_array,
    serialize_ndarray,
)
from container_models.scan_image import ScanImage


class MarkType(StrEnum):
    # Impression marks
    BREECH_FACE_IMPRESSION = auto()
    CHAMBER_IMPRESSION = auto()
    EJECTOR_IMPRESSION = auto()
    EXTRACTOR_IMPRESSION = auto()
    FIRING_PIN_IMPRESSION = auto()

    # Striation marks
    APERTURE_SHEAR_STRIATION = auto()
    BULLET_GEA_STRIATION = auto()
    BULLET_LEA_STRIATION = auto()
    CHAMBER_STRIATION = auto()
    EJECTOR_STRIATION = auto()
    EJECTOR_PORT_STRIATION = auto()
    EXTRACTOR_STRIATION = auto()
    FIRING_PIN_DRAG_STRIATION = auto()

    def is_impression(self) -> bool:
        return "IMPRESSION" in self.name

    def is_striation(self) -> bool:
        return "STRIATION" in self.name

    @property
    def scale(self) -> float:
        if self == MarkType.BREECH_FACE_IMPRESSION:
            return 3.5e-6
        return 1.5e-6


def validate_rectangle_shape(arr: NDArray[float64]) -> NDArray[float64]:
    """Validate that array has shape (4, 2)"""
    if arr.shape != (4, 2):
        raise ValueError(f"Rectangle must have shape (4, 2), got {arr.shape}")
    return arr


RectangularCrop = Annotated[
    NDArray[float64],
    BeforeValidator(partial(coerce_to_array, float64)),
    AfterValidator(partial(validate_rectangle_shape)),
    PlainSerializer(serialize_ndarray),
]


class Mark(ConfigBaseModel):
    """
    Representation of a mark (impression or striation)
    """

    scan_image: ScanImage
    mark_type: MarkType
    meta_data: dict = Field(default_factory=dict)
    _center: tuple[float, float] | None = None

    @computed_field
    @property
    def center(self) -> tuple[float, float]:
        """
        Center point of the mark in image coordinates.

        Returns the center as (x, y) where x is the horizontal position
        (column) and y is the vertical position (row). If no explicit
        center has been set, computes it as the geometric center of the
        scan image.

        :returns: Center coordinates as (x, y),
        """
        if self._center is not None:
            return self._center
        data = self.scan_image.data
        return data.shape[1] / 2, data.shape[0] / 2

    def export(self) -> str:
        """Export the `Mark` meta-data fields as a JSON string."""
        data = {
            "mark_type": self.mark_type.name,
            "center": self.center,
            "scale_x": self.scan_image.scale_x,
            "scale_y": self.scan_image.scale_y,
            "meta_data": self.meta_data,
        }
        return json.dumps(data, indent=4)
